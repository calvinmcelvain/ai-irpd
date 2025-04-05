import logging
from typing import List
from time import sleep

from utils import load_json_n_validate
from models.irpd.managers import ConfigManager, OutputManager
from models.irpd.output_processer import OutputProcesser
from models.irpd.test_prompts import TestPrompts
from models.irpd.outputs import TestOutput, TestMeta
from models.batch_output import BatchOut
from models.irpd.outputs import StageOutput


log = logging.getLogger(__name__)



class TestRunner:
    def __init__(
        self,
        config_manager: ConfigManager,
        output_manager: OutputManager,
        print_response: bool = False
    ):
        self.config_manager = config_manager
        self.output_manger = output_manager
        self.print_response = print_response
        self.processor = OutputProcesser
        
        self.stages = config_manager.config.stages
        self.test_path = config_manager.config.test_path
    
    def _prompt_id(self, stage: str, subset: str, n: int, user: object):
        prompt_id = f"{n}-{subset}"
        if stage in {"2", "3"}:
            prompt_id += f"-{user["window_number"]}"
        return prompt_id
        
    def _compose_prompts(self, stage_outputs: List[StageOutput]):
        aggregated_prompts = []
        for stage_output in stage_outputs:
            if not stage_output.complete:
                config = stage_output.stage_config
                stage = config.stage_name
                subset = config.subset
                n = config.replication
                
                test_prompts = TestPrompts(config, self.output_manger)
                prompts = [
                    (self._prompt_id(stage, subset, n, prompt.user), prompt)
                    for prompt in test_prompts.get_prompts()
                ]
                aggregated_prompts.extend(prompts)
        return aggregated_prompts
    
    def _run_batch(self, stage_outputs: List[StageOutput]):
        stage_name = stage_outputs[0].stage_name
        llm_str = stage_outputs[0].llm_str
        llm_instance = stage_outputs[0].stage_config.llm_instance
        schema = stage_outputs[0].stage_config.schema
        meta_path = stage_outputs[0].stage_config.meta_path
        
        prompts = self._compose_prompts(stage_outputs)
        
        if not prompts:
            return True
        
        if meta_path.exists():
            meta: TestMeta = load_json_n_validate(meta_path, TestMeta)
            if meta.stages[stage_name].batch_id:
                log.info(
                    f"Batch not ready for {llm_str}. Skipping stages."
                )
                return False
        
        batch_file_path = self.test_path / "_batches" / f"stage_1_{llm_str}.jsonl"
        batch_id = llm_instance.request_batch(prompts, schema, batch_file_path)
        self.processor(stage_outputs).write_meta()
        
        retries = 0
        while retries < 6:
            batch_out = llm_instance.retreive_batch(batch_id, schema, batch_file_path)
            if isinstance(batch_out, BatchOut):
                self.output_manger.store_batch(llm_str, stage_name, batch_out)
                return True
            else:
                sleep(10)
        return False
    
    def _run_completions(self, stage_outputs: List[StageOutput]):
        llm_instance = stage_outputs[0].stage_config.llm_instance
        schema = stage_outputs[0].stage_config.schema
        
        for stage_output in stage_outputs:
            prompts = self._compose_prompts(list(stage_output))
            
            if not prompts:
                continue
            
            for p in prompts:
                _, prompt = p
                
                output = llm_instance.request(prompt, schema)
                
                self.output_manger.store_completion(stage_output.stage_config, output)
        return True
    
    def run(self):
        for llm_str in self.llms:
            test_output: TestOutput = self.output_manger.test_outputs[llm_str]
            
            if test_output.complete:
                continue
            
            for stage_name in self.stages:
                stage_outputs: List[StageOutput] = self.output_manger.retrieve(
                    llm_str=llm_str, stage_name=stage_name
                )
                if stage_outputs[0].stage_config.batches:
                    complete = self._run_batch(stage_outputs)
                else:
                    complete = self._run_completions(stage_outputs)
                if not complete: break
        return self.output_manger
