import logging
from typing import List
from time import sleep

from utils import load_json_n_validate, to_list, create_directory
from models.prompts import Prompts
from models.llms.base_llm import BaseLLM
from models.irpd.output_processer import OutputProcesser
from models.irpd.test_prompts import TestPrompts
from models.irpd.test_outputs import TestOutput, TestMeta
from models.batch_output import BatchOut
from models.irpd.test_outputs import StageOutput
from models.irpd.test_config import TestConfig
from models.irpd.config_manager import ConfigManager
from models.irpd.output_manager import OutputManager


log = logging.getLogger(__name__)



class TestRunner:
    def __init__(
        self,
        test_config: TestConfig,
        output_manager: OutputManager,
        print_response: bool = False
    ):
        self.config_manager = ConfigManager(test_config)
        self.output_manger = output_manager
        self.print_response = print_response
        self.processor = OutputProcesser
        self.generate_llm_instance = self.config_manager.generate_llm_instance
        
        self.test_config = test_config
        self.llms = test_config.llms
        self.stages = test_config.stages
        self.test_path = test_config.test_path
    
    def _prompt_id(self, stage: str, subset: str, n: int, user: object):
        prompt_id = f"{n}-{subset}"
        if stage in {"2", "3"}:
            prompt_id += f"-{user["window_number"]}"
        return prompt_id
        
    def _compose_prompts(self, stage_outputs: List[StageOutput]):
        aggregated_prompts = []
        for stage_output in stage_outputs:
            if not stage_output.complete:
                llm_str = stage_output.llm_str
                stage = stage_output.stage_name
                subset = stage_output.subset
                n = stage_output.replication
                
                test_prompts = TestPrompts(
                    test_config=self.test_config,
                    llm_str=llm_str,
                    stage_name=stage,
                    replication=n,
                    subset=subset,
                    output_manager=self.output_manger
                )
                for prompt in test_prompts.get_prompts():
                    string_prompt = Prompts(system=str(prompt.system), user=str(prompt.user))
                    aggregated_prompts.append(
                        (self._prompt_id(stage, subset, n, prompt.user), string_prompt)
                    )
        return aggregated_prompts
    
    def _run_batch(self, stage_outputs: List[StageOutput], llm_instance: BaseLLM):
        if self.output_manger._check_output_set_completeness(stage_outputs):
            return True
        
        stage_name = stage_outputs[0].stage_name
        llm_str = stage_outputs[0].llm_str
        schema = self.output_manger.schemas[stage_name]
        meta_path = self.config_manager.generate_meta_path(1, llm_str)
        
        agg_prompts = self._compose_prompts(stage_outputs)
        
        if meta_path.exists():
            meta: TestMeta = load_json_n_validate(meta_path, TestMeta)
            if stage_name in meta.stages.keys():     
                if meta.stages[stage_name].batch_id:
                    log.info(
                        f"Batch not ready for {llm_str}. Skipping stages."
                    )
                    return False
        
        batches_path = self.test_path / "_batches"
        if not batches_path.exists(): create_directory(batches_path)
        batch_file_path = batches_path / f"stage_{stage_name}_{llm_str}.jsonl"
        
        batch_id = llm_instance.request_batch(agg_prompts, schema, batch_file_path)
        
        log.info(
            f"\n Requesting batch for:"
            f"\n\t llm: {llm_str}"
            f"\n\t stage: {stage_name}"
            f"\n\t replications: {self.test_config.total_replications}"
            f"\n\t batch_id: {batch_id}"
        )
        
        for stage_output in stage_outputs:
            stage_output.batch_id = batch_id
            stage_output.batch_path = batch_file_path
        self.processor(stage_outputs, self.config_manager).write_meta()
        
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
            
            if retries < 6:
                time_to_wait = 10 + retries * 10
                log.info(f"Waiting {time_to_wait} seconds.")
                sleep(time_to_wait)
                retries += 1
            else:
                log.warning(f"Retries exhausted.")
                break
        return False
    
    def _run_completions(self, stage_outputs: List[StageOutput], llm_instance: BaseLLM):
        stage_name = stage_outputs[0].stage_name
        schema = self.output_manger.schemas[stage_name]
        
        for stage_output in stage_outputs:
            llm_str = stage_output.llm_str
            replication = stage_output.replication
            subset = stage_output.subset
            
            agg_prompts = self._compose_prompts(to_list(stage_output))
            
            if len(agg_prompts) == len(stage_output.outputs) or not agg_prompts:
                stage_output.complete = True
                self.output_manger.store_completion(stage_output, stage_output.outputs)
                self.output_manger.write_output([stage_output])
                continue
            else:
                stage_output.complete = False
            
            outputs = []
            for p in agg_prompts:
                _, prompts = p
                
                log.info(
                    f"\n Requesting completion for:"
                    f"\n\t llm: {llm_str}"
                    f"\n\t replicate: {replication} of {self.test_config.total_replications}"
                    f"\n\t stage: {stage_name}"
                    f"\n\t subset: {subset}"
                    f"\n\t prompt: {len(outputs) + 1} of {len(agg_prompts)}"
                )
                outputs.append(llm_instance.request(prompts, schema))
            
            stage_output.outputs = to_list(outputs)
            
            self.output_manger.write_output([stage_output])
            stage_output.complete = True
            
            self.output_manger.store_completion(stage_output, outputs)
        self.output_manger.write_output(stage_outputs)
        return True
    
    def run(self):
        for llm_str in self.llms:
            test_output: TestOutput = self.output_manger.test_outputs[llm_str]
            llm_instance: BaseLLM = self.generate_llm_instance(llm_str, self.print_response)
            
            if test_output.complete:
                continue
            
            for stage_name in self.stages:
                stage_outputs: List[StageOutput] = self.output_manger.retrieve(
                    llm_str=llm_str, stage_name=stage_name
                )
                if self.test_config.batches and llm_instance.batches:
                    if stage_outputs[0].batch_id is None:
                        complete = self._run_batch(stage_outputs, llm_instance)
                    else:
                        complete = True
                else:
                    complete = self._run_completions(stage_outputs, llm_instance)
                if not complete: break
        return self.output_manger
