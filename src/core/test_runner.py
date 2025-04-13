"""
Test runner module.

Contains the functional TestRunner model.
"""
import logging
from typing import List
from pathlib import Path
from time import sleep

from helpers.utils import to_list, create_directory
from core.llms.clients.base import BaseLLM
from core.llms.llm_models import LLMModel
from core.output_processer import OutputProcesser
from core.irpd_prompts import IRPDPrompts
from core.managers.output_manager import OutputManager
from types.stage_output import  StageOutput
from types.batch_output import BatchOut
from types.irpd_config import IRPDConfig
from types.prompts import Prompts


log = logging.getLogger(__name__)



class TestRunner:
    """
    TestRunner model.
    
    Runs a test config. Has methods to run completion and batch. Main `run` 
    method returns the complete OutputManager model.
    """
    def __init__(
        self,
        irpd_config: IRPDConfig,
        print_response: bool = False
    ):
        self.output_manger = OutputManager(irpd_config)
        self.print_response = print_response
        
        self.irpd_config = irpd_config
        self.llms = irpd_config.llms
        self.stages = irpd_config.stages
        self.test_path = Path(irpd_config.test_path)
    
    def _generate_llm_instance(self, llm_str: str, print_reponse: bool = False):
        """
        Returns the LLM model instance from the /llms package.
        """
        return getattr(LLMModel, llm_str).get_llm_instance(self.llm_config, print_reponse)
    
    def _prompt_id(self, stage: str, subset: str, n: int, user: object):
        """
        Generates the prompt ID.
        """
        prompt_id = f"{n}-{subset}"
        if stage in {"2", "3"}:
            prompt_id += f"-{user["window_number"]}"
        return prompt_id
        
    def _compose_prompts(self, stage_outputs: List[StageOutput]):
        """
        Returns all prompts for a given stage. For batch completions, this is 
        the total prompts for a stage for every the replication.
        """
        aggregated_prompts = []
        for stage_output in stage_outputs:
            # Skipping prompt if StageOutput object is complete.
            if not stage_output.complete:
                llm_str = stage_output.llm_str
                stage = stage_output.stage_name
                subset = stage_output.subset
                n = stage_output.replication
                
                test_prompts = IRPDPrompts(
                    llm_str=llm_str,
                    stage_name=stage,
                    replication=n,
                    subset=subset,
                    output_manager=self.output_manger
                )
                
                # Creating a tuple for each prompt, where the first element is 
                # the prompt id, and the second is the Prompts object.
                for prompt in test_prompts.get_prompts():
                    string_prompt = Prompts(system=str(prompt.system), user=str(prompt.user))
                    aggregated_prompts.append(
                        (self._prompt_id(stage, subset, n, prompt.user), string_prompt)
                    )
        
        return aggregated_prompts
    
    def _run_batch(
        self,
        stage_name: str,
        stage_outputs: List[StageOutput],
        llm_instance: BaseLLM
    ):
        """
        Runs the batch request for a given stage and LLM.
        
        One batch is the requests for the stage for every replication.
        """
        # Structured output schema.
        schema = self.output_manger.schemas[stage_name]
        llm_str = stage_outputs[0].llm_str
        
        # Getting prompts.
        agg_prompts = self._compose_prompts(stage_outputs)
        
        # Composing a batch path in the test directory /_batches. Identified by 
        # the LLM & stage.
        batches_path = self.test_path / "_batches"
        if not batches_path.exists(): create_directory(batches_path)
        batch_file_path = batches_path / f"stage_{stage_name}_{llm_str}.jsonl"
        
        # Requesting batch if the batch hasn't been requested yet.
        if not (stage_outputs[0].batch_id and stage_outputs[0].batch_id):
            # Requesting batch.
            batch_id = llm_instance.request_batch(agg_prompts, schema, batch_file_path)
            
            log.info(
                f"\n Requesting batch for:"
                f"\n\t config: {self.irpd_config.id}"
                f"\n\t case: {self.irpd_config.case}"
                f"\n\t llm: {llm_str}"
                f"\n\t stage: {stage_name}"
                f"\n\t replications: {self.irpd_config.total_replications}"
                f"\n\t batch_id: {batch_id}"
            )
            
            # Storing the batch ID and path.
            for stage_output in stage_outputs:
                stage_output.batch_id = batch_id
                stage_output.batch_path = batch_file_path
            
            # Writing meta so that the ID and path are defined.
            OutputProcesser(stage_outputs, self.irpd_config).write_meta()
        
        # Retrieving batch
        retries = 0
        while retries < 6:
            batch_out = llm_instance.retreive_batch(batch_id, schema, batch_file_path)
            
            if isinstance(batch_out, BatchOut):
                self.output_manger.store_batch(
                    llm_str, stage_name, batch_out, batch_file_path
                )
                return True
            elif batch_out == "failed":
                break
            
            # If batch not ready, wait 10 seconds + 10 seconds for every retry 
            # after the first.
            retries += 1
            if retries < 6:
                time_to_wait = 10 + (retries - 1) * 10
                log.info(f"Waiting {time_to_wait} seconds.")
                sleep(time_to_wait)
            else:
                log.warning(f"Retries exhausted.")
                break
        return False
    
    def _run_completions(
        self,
        stage_name: str,
        stage_outputs: List[StageOutput],
        llm_instance: BaseLLM
    ):
        """
        Requests a chat completion for each StageOutput object for a given stage.
        
        Includes all StageOutputs across all replications.
        """
        # Structured output schema.
        schema = self.output_manger.schemas[stage_name]
        
        for stage_output in stage_outputs:
            llm_str = stage_output.llm_str
            replication = stage_output.replication
            subset = stage_output.subset
            
            agg_prompts = self._compose_prompts(to_list(stage_output))
            
            # Revalidating whether a StageOutput is complete. Because when 
            # storing completed requests in the initialization of the 
            # OutputManager, its not necessarily true that it was all outputs 
            # (e.g., could have missed a subset or summary classifications).
            if len(agg_prompts) == len(stage_output.outputs) or not agg_prompts:
                stage_output.complete = True
                self.output_manger.store_completion(stage_output, stage_output.outputs)
                self.output_manger.write_output(stage_output)
                continue
            else:
                stage_output.complete = False
            
            # Requests made for each prompt (accounts for iterative stages).
            outputs = []
            for p in agg_prompts:
                _, prompts = p
                
                log.info(
                    f"\n Requesting completion for:"
                    f"\n\t config: {self.irpd_config.id}"
                    f"\n\t case: {self.irpd_config.case}"
                    f"\n\t llm: {llm_str}"
                    f"\n\t replicate: {replication} of {self.irpd_config.total_replications}"
                    f"\n\t stage: {stage_name}"
                    f"\n\t subset: {subset}"
                    f"\n\t prompt: {len(outputs) + 1} of {len(agg_prompts)}"
                )
                outputs.append(llm_instance.request(prompts, schema))
            
            # Storing & writing outputs.
            stage_output.outputs = outputs
            self.output_manger.store_completion(stage_output)
        return True
    
    def run(self):
        """
        Runs each stage of a IRPDConfig.
        """
        # Should probably make this an async method.
        for llm_str in self.llms:
            llm_instance: BaseLLM = self._generate_llm_instance(
                llm_str, self.print_response
            )
            
            # Skipping if the test
            if self.output_manger.check_irpd_test_completion(llm_str):
                continue
            
            for stage_name in self.stages:
                # Getting all StageOutputs for a given LLM and stage.
                stage_outputs: List[StageOutput] = self.output_manger.retrieve(
                    llm_str=llm_str, stage_name=stage_name
                )
                
                # Checking whether the stage has already been complete for the 
                # given LLM and stage.
                if not all(output.complete for output in stage_outputs):
                    # Just because a IRPDConfig is specified for `batches`, not 
                    # all LLMs support batches. This adjusts for that.
                    if self.irpd_config.batches and llm_instance.batches:
                        complete = self._run_batch(
                            stage_outputs, stage_name, llm_instance
                        )
                    else:
                        if self.irpd_config.batches: log.info(
                            f"Note that {llm_str} does not support batches."
                        )
                        complete = self._run_completions(
                            stage_outputs, stage_name, llm_instance
                        )
                    if not complete: break
        
        return self.output_manger
