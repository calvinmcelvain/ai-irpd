import logging
import asyncio
from itertools import product
from time import sleep
from pathlib import Path

from utils import lazy_import
from models.irpd.test_configs import TestConfig
from models.irpd.test_prompts import TestPrompts
from models.irpd.outputs import TestOutput
from models.batch_output import BatchOut
from models.request_output import RequestOut
from models.llm_model import LLMModel
from models.llms.base_llm import BaseLLM
from models.irpd.output_processer import OutputProcesser


log = logging.getLogger(__name__)



class TestRunner:
    def __init__(
        self,
        config: TestConfig,
        output: TestOutput,
        print_response: bool = False
    ):
        self.config = config
        self.case = config.case
        self.cases = config.case.split("_")
        self.stages = config.stages
        self.batch_request = config.batches
        self.llm_config = config.llm_config
        self.test_path = config.test_path
        self.total_replications = config.total_replications
        self.replications = config.total_replications
        self.llms = config.llms
        self.output = output
        self.output_processor = OutputProcesser(config)
        self.print_response = print_response
    
    @staticmethod
    def _get_instance_types(case: str):
        if case in {"uni", "uniresp"}:
            return ["ucoop", "udef"]
        return ["coop", "def"]
    
    @staticmethod
    def _get_stage_schema(stage: str):
        return lazy_import("models.irpd.schemas", f"Stage{stage}Schema")
    
    def _get_subsets(self, stage: str):
        if stage in {"1c", "2", "3"}:
            return ["full"]
        subsets = [
            f"{c}_{i}" for c in self.cases
            for i in self._get_instance_types(c)
        ]
        return subsets + ["full"]
        
    def _generate_subpath(self, N: int, llm_str: str):
        subpath = self.test_path
        if len(self.llms) > 1: subpath = subpath / llm_str
        if self.total_replications > 1: subpath = subpath / f"replication_{N}"
        return Path(subpath)
    
    def _generate_llm_instance(self, llm: str):
        return getattr(LLMModel, llm).get_llm_instance(
            self.llm_config, self.print_response
        )
    
    def _prompt_id(self, stage: str, subset: str, n: int, user: object):
        prompt_id = f"{n}-{subset}"
        if stage in {"2", "3"}:
            prompt_id += f"-{user["window_number"]}"
        return prompt_id
        
    def _compose_prompts(self, stage: str, llm: str):
        subsets = self._get_subsets(stage)
        aggregated_prompts = []
        for n, subset in product(range(1, self.total_replications + 1), subsets):
            subpath = self._generate_subpath(n, llm)
            complete = self.output.check_output(subpath, llm, n, stage)
            if not complete:
                test_prompts = TestPrompts(stage, n, subset, llm, self.config, self.output)
                prompts = [
                    (self._prompt_id(stage, subset, n, prompt.user), prompt)
                    for prompt in test_prompts.get_prompts()
                ]
                aggregated_prompts.extend(prompts)
        return aggregated_prompts
    
    async def _run_batch(self, stage: str, llm: str):
        llm_instance: BaseLLM = self._generate_llm_instance(llm)
        prompts = self._compose_prompts(stage, llm)
        schema = self._get_stage_schema(stage)
        
        if prompts:
            batch_file_path = self.test_path / "_batches" / f"stage_1_{llm}.jsonl"
            batch_id = llm_instance.request_batch(prompts, schema, batch_file_path)
            
            batch_complete = False
            while not batch_complete:
                batch_response = llm_instance.retreive_batch(batch_id, schema, batch_file_path)
                if isinstance(batch_response, BatchOut):
                    batch_complete = True
                    for r in batch_response.responses:
                        response_id  = r.response_id
                        response = r.response
                
                        id_split = response_id.split("-")
                        replication = id_split[0]
                        subset = id_split[1]
                        
                        self.output.store(stage, llm, replication, subset, response)
                else:
                    sleep(10)
        return None
    
    async def _run_completions(self, stage: str, llm: str):
        llm_instance: BaseLLM = self._generate_llm_instance(llm)
        prompts = self._compose_prompts(stage, llm)
        schema = self._get_stage_schema(stage)
        
        if prompts:
            for i in prompts:
                prompt_id, prompt = i
                
                id_split = prompt_id.split("-")
                replication = id_split[0]
                subset = id_split[1]
                
                output: RequestOut = llm_instance.request(prompt, schema)
                
                self.output.store(stage, llm, replication, subset, output)
        return None
    
    async def run(self):
        for stage in self.stages:
            tasks = []
            for llm in self.llms:
                if self.batch_request:
                    tasks.append(self._run_batch(stage, llm))
                else:
                    tasks.append(self._run_completions(stage, llm))
            await asyncio.gather(*tasks)
        return self.output
