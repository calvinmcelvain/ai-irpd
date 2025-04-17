"""
Test runner module.

Contains the functional TestRunner model.
"""
import logging
from typing import List
from time import sleep
from tqdm import tqdm

from helpers.utils import create_directory
from core.functions import requestout_to_irpdout
from core.foundation import FoundationalModel
from core.llms.clients.base_llm import BaseLLM
from core.output_manager import OutputManager
from _types.test_output import TestOutput
from _types.batch_output import BatchOut
from _types.irpd_config import IRPDConfig


log = logging.getLogger("app")



class TestRunner(FoundationalModel):
    """
    TestRunner model.
    
    Runs a test config. Has methods to run completion and batch. Main `run` 
    method returns the complete OutputManager model.
    """
    def __init__(self, irpd_config: IRPDConfig, print_response: bool):
        super().__init__(irpd_config, print_response)
        
        self.output_manger = OutputManager(irpd_config)
        self.prompt_composer = self.output_manger.prompt_composer
    
    def _run_batch(
        self,
        stage_name: str,
        test_outputs: List[TestOutput],
        llm_instance: BaseLLM
    ):
        # Structured output schema.
        schema = self.schemas[stage_name]
        llm_str = test_outputs[0].llm_str
        
        log.info(
            f"Starting batch run for stage '{stage_name}' with LLM '{llm_str}'.")
        
        # Getting prompts.
        agg_prompts = self.prompt_composer.get_prompts(test_outputs, stage_name)
        
        # Composing a batch path in the test directory /_batches. Identified by 
        # the LLM & stage.
        batches_path = self.test_path / "_batches"
        create_directory(batches_path)
        batch_file_path = batches_path / f"stage_{stage_name}_{llm_str}.jsonl"
        
        # Requesting batch if the batch hasn't been requested yet.
        if not batch_file_path.exists():
            log.info(
                f"Batch file '{batch_file_path}' does not exist. Requesting"
                " batch."
            )
            
            # Requesting batch.
            batch_id = llm_instance.request_batch(
                agg_prompts, schema, batch_file_path)
            
            with tqdm(total=1, desc="Requesting batch", unit="batch") as progress_bar:
                progress_bar.set_postfix({
                    "config": self.irpd_config.id,
                    "case": self.case,
                    "llm": llm_str,
                    "stage": stage_name,
                    "replications": self.total_replications,
                    "batch_id": batch_id
                })
                progress_bar.update(1)
            
            log.info(
                f"Batch requested with ID '{batch_id}' for stage '{stage_name}'.")
            
            # Storing batch ID in meta
            test_outputs[0].meta.stages[stage_name].batch_id = batch_id
            test_outputs[0].meta.stages[stage_name].batch_id = batches_path
            self.output_manger.output_writer.write_meta(test_outputs[0])
        
        # Retrieving batch
        retries = 0
        log.info(
            f"Retrieving batch for stage '{stage_name}' with a maximum of"
            " 6 retries."
        )
        with tqdm(total=6, desc="Retrieving batch", unit="retry") as progress_bar:
            while retries < 6:
                batch_out = llm_instance.retreive_batch(batch_id, schema, batch_file_path)
                
                if isinstance(batch_out, BatchOut):
                    log.info(
                        f"Batch successfully retrieved for stage '{stage_name}'.")
                    
                    self.output_manger.store_batch(llm_str, stage_name, batch_out)
                    return True
                elif batch_out == "failed":
                    log.error(f"Batch retrieval failed for stage '{stage_name}'.")
                    break
                
                # If batch not ready, wait 10 seconds + 10 seconds for every retry 
                # after the first.
                retries += 1
                progress_bar.update(1)
                if retries < 6:
                    time_to_wait = 10 + (retries - 1) * 10
                    progress_bar.set_postfix({"waiting_time": f"{time_to_wait} seconds"})
                    sleep(time_to_wait)
                else:
                    progress_bar.set_postfix({"status": "Retries exhausted"})
                    break
        return False
    
    def _run_completions(
        self,
        stage_name: str,
        test_outputs: List[TestOutput],
        llm_instance: BaseLLM
    ):
        # Structured output schema.
        schema = self.schemas[stage_name]
        llm_str = test_outputs[0].llm_str
        
        log.info(
            f"Starting completions for stage '{stage_name}' with LLM {llm_str}")
        
        # Getting prompts.
        agg_prompts = self.prompt_composer.get_prompts(test_outputs, stage_name)
        
        # Requests made for each prompt (accounts for iterative stages).
        progress_bar = tqdm(
            total=len(agg_prompts),
            desc=f"Processing completions for {stage_name}",
            unit="prompt"
        )
        for idx, (prompt_id, prompt) in enumerate(agg_prompts, start=1):
            id_list = prompt_id.split("-")
            n = int(id_list[0])
            subset = id_list[1]
            
            stage_output = self.output_manger.retrieve(llm_str, n, stage_name)
            subset_path = stage_output.stage_path / subset
            
            progress_bar.set_postfix({
                "config": self.irpd_config.id,
                "case": self.case,
                "llm": llm_str,
                "replicate": f"{n}/{self.total_replications}",
                "stage": stage_name,
                "subset": subset,
                "prompt": f"{idx}/{len(agg_prompts)}"
            })
            
            output = llm_instance.request(prompt, schema)
            irpd_output = requestout_to_irpdout(
                stage_name, subset, subset_path, output)
            
            progress_bar.update(1)
            self.output_manger.store_output(llm_str, n, stage_name, irpd_output)
        progress_bar.close()
        log.info(f"All completions processed for stage '{stage_name}'.")
        return True
    
    def run(self):
        log.info("Starting TestRunner execution.")
        # Should probably make this an async method.
        for llm_str, llm_instance in self.llm_instances.items():
            log.info(f"Processing LLM '{llm_str}'.")
            # Getting all TestOutputs for a given LLM.
            test_outputs = self.output_manger.retrieve(llm_str)
            
            # Skipping if all outputs complete.
            if all(m.complete for m in test_outputs): continue
            
            for stage_name in self.stages:
                log.info(f"Processing stage '{stage_name}' for LLM '{llm_str}'.")
                
                # Skipping if all outputs complete.
                if all(m.stage_outputs[stage_name].complete for m in test_outputs): continue
                
                # Some LLMs don't support batches.
                if self.batches and llm_instance.batches:
                    complete = self._run_batch(
                        stage_name, test_outputs, llm_instance
                    )
                else:
                    if self.batches: log.info(
                        f"Note that {llm_str} does not support batches."
                    )
                    
                    complete = self._run_completions(
                        stage_name, test_outputs, llm_instance
                    )
                
                # Breaks if batches couldn't be retrieved in 6 trys.
                if not complete:
                    log.error(
                        f"Stage '{stage_name}' could not be completed for LLM"
                        f" '{llm_str}'. Aborting."
                    )
                    break
        log.info("TestRunner execution completed.")
        return self.output_manger.outputs
