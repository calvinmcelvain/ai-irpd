import logging
from time import sleep
from pathlib import Path
from itertools import product
from typing import List, Optional, Union, Dict

from utils import check_directories, load_json_n_validate, lazy_import, to_list
from models.llm_model import LLMModel
from models.request_output import RequestOut
from models.batch_output import BatchOut
from models.irpd.test_configs import TestConfig
from models.irpd.outputs import TestOutput, StageOutput, TestMeta
from models.irpd.output_processer import OutputProcesser


log = logging.getLogger(__name__)



class ConfigManager:
    def __init__(self, test_config: TestConfig):
        self.config = test_config
        self.test_path = test_config.test_path
        self.llms = test_config.llms
        self.llm_config = test_config.llm_config
        self.total_replications = test_config.total_replications
    
    def generate_subpath(self, n: int, llm_str: str):
        subpath = self.test_path
        if len(self.llms) > 1: subpath = subpath / llm_str
        if self.total_replications > 1: subpath = subpath / f"replication_{n}"
        return Path(subpath)
    
    def generate_meta_path(self, n: int, llm_str: str):
        subpath = self.generate_subpath(n, llm_str)
        return subpath / "_test_meta.json"
    
    def get_subsets(self, stage_name: str):
        subsets = ["full"]
        if stage_name in {"1", "1r"}:
            prod = product(self.config.cases, self.config.instance_types)
            subsets += [f"{c}_{i}" for c, i in prod]
        return subsets
        


class OutputManager:
    def __init__(self, test_config: TestConfig):
        self.config_manager = ConfigManager(test_config)
        self.test_config = test_config
        self.processor = OutputProcesser
        self.test_path = test_config.test_path
        self.total_replications = test_config.total_replications
        self.stages = test_config.stages
        self.llms = test_config.llms
        self.llm_config = test_config.llm_config
        self.schemas = {
            stage: lazy_import("models.irpd.schemas", f"Stage{stage}Schema")
            for stage in self.stages
        }
        
        self.test_outputs = self._initialize_test_outputs()
        
        self._check_test_directory()
        self._check_batch()
        self._check_completeness()
        
    def _initialize_test_outputs(self):
        test_outputs = {}
        for llm_str in self.test_config.llms:
            test_output = TestOutput(llm_str=llm_str,)
            for n in range(1, self.total_replications + 1):
                for stage in self.stages:
                    subsets = self.config_manager.get_subsets(stage)
                    for subset in subsets:
                        test_output.stage_outputs.append(StageOutput(
                            stage_name=stage,
                            subset=subset,
                            llm_str=llm_str,
                            replication=n
                        ))
            test_outputs[llm_str] = test_output
        return test_outputs
    
    def _check_completeness(self, llm_str: str = None):
        llm_str = to_list(llm_str) if llm_str else self.llms
        for llm in llm_str:
            self.test_outputs[llm].test_complete()
            if self.test_outputs[llm].complete:
                log.info(f"All outputs for {llm} are complete.")
        return None
    
    def _check_test_directory(self):
        for llm_str in self.llms:
            for stage_output in self.test_outputs[llm_str].stage_outputs:
                stage = stage_output.stage_name
                sub_path = self.config_manager.generate_subpath(stage_output.replication, stage_output.llm_str)
                responses_path = sub_path / stage_output.stage_name / stage_output.subset / "responses"
                prompts_path = sub_path / stage_output.stage_name / stage_output.subset / "prompts"
                
                if not check_directories([responses_path, prompts_path]):
                    continue
                
                outputs = [
                    RequestOut(parsed=load_json_n_validate(path, self.schemas[stage]))
                    for path in responses_path.iterdir()
                ]
                
                self.store_completion(stage_output, outputs)
        
        return None
    
    def _check_batch(self):
        for llm_str in self.llms:
            llm = self._generate_llm_instance(llm_str)
            test_output = self.test_outputs[llm_str]
            if not test_output.complete:
                for stage in self.test_config.stages:
                    meta_path = self.config_manager.generate_meta_path(llm_str, 1)
                    if self.test_config.batches and meta_path.exists():
                        meta: TestMeta = load_json_n_validate(meta_path, TestMeta)
                        batch_id = meta.stages[stage].batch_id
                        batch_path = meta.stages[stage].batch_path
                        
                        if batch_id is None:
                            continue
                        
                        retries = 0
                        while retries > 6:
                            schema = self.schemas[stage]
                            batch_out = llm.retreive_batch(batch_id, schema, batch_path)
                            
                            if isinstance(batch_out, BatchOut):
                                self.store_batch(llm_str, stage, batch_out)
                                break
                            
                            if retries < 6:
                                log.info(f"Batch is {batch_out}; Waiting 30 seconds.")
                                sleep(15)
                                retries += 1
                            else:
                                log.warning(f"Batch is {batch_out}; Retries exhausted.")
                                break
        return None
    
    def _generate_llm_instance(self, llm_str: str):
        return getattr(LLMModel, llm_str).get_llm_instance(self.llm_config)
    
    def _get_output_index(self, stage_output: StageOutput):
        return self.test_outputs[stage_output.llm_str].index(stage_output)
    
    def _check_stage_completion(self, stage_output: StageOutput):
        stage_name = stage_output.stage_name
        llm_str = stage_output.llm_str
        n = stage_output.replication
        
        stage_outputs = self.retrieve(llm_str, n, stage_name)
        stage_complete = all(
            output.complete for output in stage_outputs
            if output.stage_name == stage_name
        )
        
        return stage_complete
    
    def _log_stored_completion(self, stage_output: StageOutput):
        config = stage_output.stage_config
        log.info(
            "\nOutputs stored successfully for:"
            f"\n\t llm: {config.llm_str}"
            f"\n\t replicate: {config.replication} / {self.config_manager.config.total_replications}"
            f"\n\t stage: {config.stage_name}"
            f"\n\t subset: {config.subset}"
            f"\n\t COMPLETE: {stage_output.complete}"
        )
        return None
    
    def retrieve(
        self,
        llm_str: Optional[str] = None,
        n: Optional[int] = None,
        stage_name: Optional[str] = None,
        subset: Optional[str] = None
    ):
        outputs: Dict[str, TestOutput] = self.test_outputs
        if llm_str:
            outputs: List[StageOutput] = outputs[llm_str].stage_outputs
            if n is not None:
                outputs: List[StageOutput] = filter(lambda output: output.replication == n, outputs)
                if stage_name:
                    outputs: List[StageOutput] = filter(lambda output: output.stage_name == stage_name, outputs.stage_outputs)
                    if subset:
                        outputs: StageOutput = filter(lambda output: output.subset == subset, outputs)
        if outputs is None:
            log.warning(
                "\nOutputs not found for:"
                f"\n\t llm: {llm_str}"
                f"\n\t replicate: {n} / {self.test_config.total_replications}"
                f"\n\t stage: {stage_name}"
                f"\n\t subset: {subset}"
            )
            return None
        return to_list(outputs)
    
    def store_completion(
        self,
        stage_output: StageOutput,
        outputs: Union[RequestOut, List[RequestOut]]
    ):
        llm_str = stage_output.llm_str
        n = stage_output.replication
        stage_name = stage_output.stage_name
        subset = stage_output.subset
        
        output = self.retrieve(llm_str, n, stage_name, subset)
        assert isinstance(output, StageOutput), "Output could not be stored."
        
        output.outputs = to_list(outputs)
        output.complete = True
        
        idx = self._get_output_index(stage_output)
        self.test_outputs[stage_output.llm_str].stage_outputs[idx] = output
        
        
        if self._check_stage_completion(stage_output):
            stage_outputs = self.retrieve(llm_str, n, stage_name)
            self.processor(to_list(stage_outputs)).process(True)
        else:
            self.processor(to_list(output)).process()
        
        self._log_stored_completion(output)
        self._check_completeness(llm_str)
        return None
    
    def store_batch(
        self,
        llm_str: str,
        stage_name: str,
        batch_out: BatchOut
    ):
        outputs = batch_out.responses
        stage_outputs: List[StageOutput] = self.retrieve(
            llm_str=llm_str,
            stage_name=stage_name
        )
        for stage_output in stage_outputs:
            assert isinstance(stage_output, StageOutput), "Output could not be stored."
            
            n  = stage_output.replication
            subset = stage_output.subset
            
            stage_output.outputs = next((
                response.response for response in outputs
                if response.response_id.startswith(f"{n}-{subset}")
            ))
            
            idx = self._get_output_index(stage_output)
            self.test_outputs[llm_str].stage_outputs[idx] = stage_output
            
            if self._check_stage_completion(stage_output):
                stage_outputs = self.retrieve(llm_str, n, stage_name)
                self.processor(to_list(stage_outputs)).process(True)
            else:
                self.processor(to_list(stage_output)).process()
            
            self._log_stored_completion(stage_output)
        self._check_completeness(llm_str)
        return None
