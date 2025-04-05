import logging
from time import sleep
from pathlib import Path
from itertools import product
from typing import List, Optional, Union

from utils import check_directories, load_json_n_validate
from models.llm_model import LLMModel
from models.request_output import RequestOut
from models.batch_output import BatchOut
from models.irpd.test_configs import TestConfig, SubConfig, StageConfig
from models.irpd.outputs import TestOutput, StageOutput, TestMeta
from models.irpd.output_processer import OutputProcesser
from models.irpd.test_prompts import TestPrompts


log = logging.getLogger(__name__)



class ConfigManager:
    def __init__(self, test_config: TestConfig):
        self.config = test_config
        self.sub_configs = self._generate_sub_configs()
        self.stage_configs = self._generate_stage_configs()
        
    def _generate_sub_configs(self):
        prod = product(self.config.llms, range(1, self.config.total_replications + 1))
        sub_configs = [
            SubConfig(
                **vars(self.test_config),
                sub_path=self._generate_subpath(n, llm_str),
                llm_str=llm_str,
                llm_instance=self._generate_llm_instance(llm_str),
                replication=n
            ) for llm_str, n in prod
        ]
        return sub_configs
    
    def _generate_stage_configs(self):
        stage_configs = []
        for sub_config in self.sub_configs:
            expected_outputs = {
                stage: TestPrompts(
                    StageConfig(**vars(sub_config), stage_name=stage)
                ).expected_outputs
                for stage in sub_config.stages
            }
            stage_configs.extend([
                StageConfig(
                    **vars(sub_config),
                    stage_name=stage_name,
                    expected_outputs=expected_outputs[stage_name],
                    subset=subset
                )
                for sub_config in self.sub_configs
                for stage_name in sub_config.stages
                for subset in self._get_subsets(stage_name)
            ])
        return stage_configs
    
    def _get_subsets(self, stage_name: str):
        subsets = ["full"]
        if stage_name in {"1", "1r"}:
            prod = product(self.config.cases, self.config.instance_types)
            subsets += [f"{c}_{i}" for c, i in prod]
        return subsets
    
    def _generate_subpath(self, n: int, llm_str: str):
        subpath = self.test_path
        if len(self.llms) > 1: subpath = subpath / llm_str
        if self.total_replications > 1: subpath = subpath / f"replication_{n}"
        return Path(subpath)
    
    def _generate_llm_instance(self, llm: str):
        return getattr(LLMModel, llm).get_llm_instance(
            self.llm_config, self.print_response
        )
    
    def retrieve(
        self,
        llm_str: Optional[str] = None,
        n: Optional[int] = None,
        stage_name: Optional[str] = None,
        subset: Optional[str] = None
    ):
        configs = self.stage_configs
        if llm_str:
            configs = filter(lambda config: config.llm_str == llm_str, configs)
        if n is not None:
            configs = filter(lambda config: config.replication == n, configs)
        if stage_name:
            configs = filter(lambda config: config.stage_name == stage_name, configs)
        if subset:
            configs = filter(lambda config: config.subset == subset, configs)
        return list(configs)
        


class OutputManager:
    def __init__(self, test_config: TestConfig):
        self.config_manager = ConfigManager(test_config)
        self.test_config = test_config
        self.processor = OutputProcesser
        
        self.test_outputs = self._initialize_test_outputs()
        
        self._check_test_directory()
        self._check_batch()
        self._check_completeness()
        
    def _initialize_test_outputs(self):
        test_outputs = {}
        for llm_str in self.test_config.llms:
            stage_configs = self.config_manager.retrieve(llm_str)
            test_outputs[llm_str] = TestOutput(
                llm_str=llm_str,
                stage_outputs=[
                    StageOutput(
                        stage_config=stage_config,
                        stage_name=stage_config.stage_name,
                        subset=stage_config.subset
                    )
                    for stage_config in stage_configs
                ]
            )
        
        return test_outputs
    
    def _check_completeness(self, llm_str: str):
        self.test_outputs[llm_str].test_complete()
        if self.test_outputs[llm_str].complete:
            log.info(f"All outputs for {llm_str} are complete.")
        return None
    
    def _check_test_directory(self):
        for test_output in self.test_outputs:
            stage_configs = self.config_manager.retrieve(test_output.llm_str)
            for stage_config in stage_configs:
                responses_path = stage_config.responses_path
                prompts_path = stage_config.prompts_path
                
                if not check_directories([responses_path, prompts_path]):
                    continue
                
                outputs = [
                    RequestOut(parsed=load_json_n_validate(path, stage_config.schema))
                    for path in responses_path.iterdir()
                ]
                
                self.store_completion(stage_config, outputs)
        
        return None
    
    def _check_batch(self):
        for test_output in self.test_outputs.values():
            llm_str = test_output.llm_str
            if not test_output.complete:
                for stage in self.test_config.stages:
                    stage_config: StageConfig = self.config_manager.retrieve(llm_str, 1, stage)[0]
                    
                    if stage_config.batches and stage_config.meta_path.exists():
                        meta: TestMeta = load_json_n_validate(stage_config.meta_path, TestMeta)
                        batch_id = meta.stages[stage_config.stage_name].batch_id
                        batch_path = meta.stages[stage_config.stage_name].batch_path
                        
                        if batch_id is None:
                            continue
                        
                        retries = 0
                        while retries > 6:
                            llm = stage_config.llm_instance
                            schema = stage_config.schema
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
    
    def _get_output_index(self, stage_config: StageConfig, output: StageOutput):
        test_output: TestOutput = self.test_outputs[stage_config.llm_str]
        output_idx = test_output.stage_outputs.index(output)
        return output_idx
    
    def _check_stage_completion(self, stage_config: StageConfig):
        stage_name = stage_config.stage_name
        llm_str = stage_config.llm_str
        n = stage_config.replication
        
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
        outputs: List[TestOutput] = self.test_outputs
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
        return list(outputs)
    
    def store_completion(
        self,
        stage_config: StageConfig,
        outputs: Union[RequestOut, List[RequestOut]]
    ):
        llm_str = stage_config.llm_str
        n = stage_config.replication
        stage_name = stage_config.stage_name
        subset = stage_config.subset
        
        output = self.retrieve(llm_str, n, stage_name, subset)
        assert isinstance(output, StageOutput), "Output could not be stored."
        
        output.outputs = list(outputs)
        
        output.complete = len(output.outputs) == stage_config.expected_outputs
        
        idx = self._get_output_index(stage_config, output)
        self.test_outputs[stage_config.llm_str].stage_outputs[idx] = output
        
        
        if self._check_stage_completion(stage_config):
            stage_outputs = self.retrieve(llm_str, n, stage_name)
            self.processor(list(stage_outputs)).process(True)
        else:
            self.processor(list(output)).process()
        
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
        stage_configs: List[StageConfig] = self.config_manager.retrieve(
            llm_str=llm_str,
            stage_name=stage_name
        )
        for stage_config in stage_configs:
            n  = stage_config.replication
            subset = stage_config.subset
            
            output = self.retrieve(llm_str, n, stage_name, subset)
            assert isinstance(output, StageOutput), "Output could not be stored."
            
            output.outputs = next((
                response.response for response in outputs
                if response.response_id.startswith(f"{n}-{subset}")
            ))
            
            idx = self._get_output_index(stage_config, output)
            self.test_outputs[llm_str].stage_outputs[idx] = output
            
            if self._check_stage_completion(stage_config):
                stage_outputs = self.retrieve(llm_str, n, stage_name)
                self.processor(list(stage_outputs)).process(True)
            else:
                self.processor(list(output)).process()
            
            self._log_stored_completion(output)
        self._check_completeness(llm_str)
        return None
