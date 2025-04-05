import logging
from pathlib import Path
from itertools import product
from typing import List, Optional, Union

from utils import check_directories, load_json_n_validate
from models.llm_model import LLMModel
from models.request_output import RequestOut
from models.batch_output import BatchOut
from models.irpd.test_configs import TestConfig, SubConfig, StageConfig
from models.irpd.outputs import TestOutput, SubOutput, StageOutput
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
        self.test_outputs = TestOutput(
            config=test_config,
            test_outputs=self._initialize_sub_outputs()
        )
        self.processor = OutputProcesser
        
    def _initialize_sub_outputs(self):
        sub_outputs = [
            SubOutput(
                sub_config=sub_config,
                llm_str=sub_config.llm_str,
                replication=sub_config.replication,
                stage_outputs=[
                    StageOutput(
                        stage_config=stage_config,
                        stage_name=stage_config.stage_name,
                        subset=stage_config.subset
                    )
                    if not self._check_output_directory(stage_config) else
                    self._check_output_directory(stage_config)
                    for stage_config in self.config_manager.retrieve(
                        llm_str=sub_config.llm_str,
                        N=sub_config.replication
                    )
                ]
            )
            for sub_config in self.config_manager.sub_configs
        ]
        return sub_outputs
    
    def _check_output_directory(self, stage_config: StageConfig):
        stage_name = stage_config.stage_name
        subset = stage_config.subset
        stage_path = stage_config.stage_path
        subset_path = stage_path / subset
        responses_path = stage_config.responses_path
        prompts_path = stage_config.prompts_path
        
        if not check_directories([responses_path, prompts_path]):
            log.info(f"No outputs found in {subset_path}")
            return None
        
        outputs = [
            RequestOut(parsed=load_json_n_validate(path, stage_config.schema))
            for path in responses_path.iterdir()
        ]
        
        expected_outputs = TestPrompts(stage_config, self).expected_outputs
        complete = len(outputs) == expected_outputs
        
        return StageOutput(stage_config, stage_name, subset, outputs, complete)
    
    def _get_output_index(self, output: StageOutput):
        sub_output: SubOutput = self.retreive(
            llm_str=output.stage_config.llm_str,
            N=output.stage_config.replication
        )
        sub_output_idx = self.test_outputs.test_outputs.index(sub_output)
        stage_output_idx = sub_output.stage_outputs.index(output)
        return sub_output_idx, stage_output_idx
    
    def _check_stage_completion(self, stage_name: str, stage_outputs: List[SubOutput]):
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
        outputs: List[SubOutput] = self.test_outputs.test_outputs
        if llm_str:
            outputs: List[SubOutput] = filter(lambda output: output.llm_str == llm_str, outputs)
            if n is not None:
                outputs: SubOutput = filter(lambda output: output.replication == n, outputs)
                if stage_name:
                    outputs: List[StageOutput] = filter(lambda output: output.stage_name == stage_name, outputs.stage_outputs)
                    if subset:
                        outputs: StageOutput = filter(lambda output: output.subset == subset, outputs)
        if outputs is None:
            log.warning(
                "\nOutputs not found for:"
                f"\n\t llm: {llm_str}"
                f"\n\t replicate: {n} / {self.config_manager.config.total_replications}"
                f"\n\t stage: {stage_name}"
                f"\n\t subset: {subset}"
            )
            return None
        return list(outputs)
    
    def store_completion(
        self,
        llm_str: str,
        n: int,
        stage_name: str,
        subset: str,
        outputs: Union[RequestOut, List[RequestOut]]
    ):
        output = self.retrieve(llm_str, n, stage_name, subset)[0]
        assert isinstance(output, StageOutput), "Output could not be stored."
        
        output.outputs = list(outputs)
        
        stage_config = self.config_manager.retrieve(
            llm_str, n, stage_name, subset
        )
        
        expected_outputs = TestPrompts(stage_config, self).expected_outputs
        output.complete = len(output.outputs) == expected_outputs
        
        stage_outputs = self.retrieve(llm_str, n, stage_name)
        stage_complete = self._check_stage_completion(stage_name, stage_outputs)
            
        self.processor(list(output)).process(stage_complete)
        
        idx_1, idx_2 = self._get_output_index(output)
        self.test_outputs.test_outputs[idx_1].stage_outputs[idx_2] = output
        
        self._log_stored_completion(output)
        return None
    
    def store_batch(
        self,
        llm_str: str,
        stage_name: str,
        batch: BatchOut
    ):
        outputs = batch.responses
        stage_configs: List[StageConfig] = self.config_manager.retrieve(llm_str)
        for stage_config in stage_configs:
            n  = stage_config.replication
            subset = stage_config.subset
            
            output = self.retrieve(llm_str, n, stage_name, subset)
            assert isinstance(output, StageOutput), "Output could not be stored."
            
            output.outputs = next((
                response.response for response in outputs
                if response.response_id == f"{n}-{subset}"
            ))
            
            expected_outputs = TestPrompts(stage_config, self).expected_outputs
            output.complete = len(output.outputs) == expected_outputs
            
            stage_outputs = self.retrieve(llm_str, n, stage_name)
            stage_complete = self._check_stage_completion(stage_name, stage_outputs)
            
            self.processor(list(output)).process(stage_complete)
            
            idx_1, idx_2 = self._get_output_index(output)
            self.test_outputs.test_outputs[idx_1].stage_outputs[idx_2] = output
            
            self._log_stored_completion(output)
        return None
