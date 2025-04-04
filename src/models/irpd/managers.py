import logging
from pathlib import Path
from itertools import product
from typing import List, Optional, Union

from utils import to_list
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
        stage_configs = [
            StageConfig(**vars(sub_config), stage_name=stage_name, subset=subset)
            for sub_config in self.sub_configs
            for stage_name in sub_config.stages
            for subset in self._get_subsets(stage_name)
        ]
        return stage_configs
    
    def _get_subsets(self, stage_name: str):
        subsets = ["full"]
        if stage_name in {"1", "1r"}:
            prod = product(self.config.cases, self.config.instance_types)
            subsets += [f"{c}_{i}" for c, i in prod]
        return subsets
    
    def _generate_subpath(self, N: int, llm_str: str):
        subpath = self.test_path
        if len(self.llms) > 1: subpath = subpath / llm_str
        if self.total_replications > 1: subpath = subpath / f"replication_{N}"
        return Path(subpath)
    
    def _generate_llm_instance(self, llm: str):
        return getattr(LLMModel, llm).get_llm_instance(
            self.llm_config, self.print_response
        )
    
    def retrieve(
        self,
        llm_str: Optional[str] = None,
        N: Optional[int] = None,
        stage_name: Optional[str] = None,
        subset: Optional[str] = None
    ):
        configs = self.stage_configs
        if llm_str:
            configs = filter(lambda config: config.llm_str == llm_str, configs)
        if N is not None:
            configs = filter(lambda config: config.replication == N, configs)
        if stage_name:
            configs = filter(lambda config: config.stage_name == stage_name, configs)
        if subset:
            configs = filter(lambda config: config.subset == subset, configs)
        return to_list(configs)
        


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
                    for stage_config in self.config_manager.retrieve(
                        llm_str=sub_config.llm_str,
                        N=sub_config.replication
                    )
                ]
            )
            for sub_config in self.config_manager.sub_configs
        ]
        return sub_outputs
    
    def _get_output_index(self, output: StageOutput):
        sub_output: SubOutput = self.retreive(
            llm_str=output.stage_config.llm_str,
            N=output.stage_config.replication
        )
        sub_output_idx = self.test_outputs.test_outputs.index(sub_output)
        stage_output_idx = sub_output.stage_outputs.index(output)
        return sub_output_idx, stage_output_idx
    
    def retrieve(
        self,
        llm_str: Optional[str] = None,
        N: Optional[int] = None,
        stage_name: Optional[str] = None,
        subset: Optional[str] = None
    ):
        outputs: List[SubOutput] = self.test_outputs.test_outputs
        if llm_str:
            outputs: List[SubOutput] = filter(lambda output: output.llm_str == llm_str, outputs)
            if N is not None:
                outputs: SubOutput = filter(lambda output: output.replication == N, outputs)
                if stage_name:
                    outputs: List[StageOutput] = filter(lambda output: output.stage_name == stage_name, outputs.stage_outputs)
                    if subset:
                        outputs: StageOutput = filter(lambda output: output.subset == subset, outputs)
        if outputs is None:
            log.warning(
                "\nOutputs not found for:"
                f"\n\t llm: {llm_str}"
                f"\n\t replicate: {N} / {self.config_manager.config.total_replications}"
                f"\n\t stage: {stage_name}"
                f"\n\t subset: {subset}"
            )
            return None
        return to_list(outputs)
    
    def store_completion(
        self,
        llm_str: str,
        N: int,
        stage_name: str,
        subset: str,
        outputs: Union[RequestOut, List[RequestOut]]
    ):
        output = self.retrieve(llm_str, N, stage_name, subset)[0]
        assert isinstance(output, StageOutput), "Output could not be stored."
        
        output.outputs = to_list(outputs)
        
        stage_config = self.config_manager.retrieve(
            llm_str, N, stage_name, subset
        )
        
        expected_outputs = TestPrompts(stage_config, self).expected_outputs
        output.complete = len(output.outputs) == expected_outputs
        
        idx_1, idx_2 = self._get_output_index(output)
        self.test_outputs.test_outputs[idx_1].stage_outputs[idx_2] = output
        log.info(
            "\nOutputs stored successfully for:"
            f"\n\t llm: {llm_str}"
            f"\n\t replicate: {N} / {self.config_manager.config.total_replications}"
            f"\n\t stage: {stage_name}"
            f"\n\t subset: {subset}"
            f"\n\t COMPLETE: {output.complete}"
        )
        
        # self.processor(output).process()
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
            replication  = stage_config.replication
            subset = stage_config.subset
            
            output = self.retrieve(llm_str, replication, stage_name, subset)
            assert isinstance(output, StageOutput), "Output could not be stored."
            
            output.outputs = next((
                response.response for response in outputs
                if response.response_id == f"{replication}-{subset}"
            ))
            
            expected_outputs = TestPrompts(stage_config, self).expected_outputs
            output.complete = len(output.outputs) == expected_outputs
            
            idx_1, idx_2 = self._get_output_index(output)
            self.test_outputs.test_outputs[idx_1].stage_outputs[idx_2] = output
            log.info(
                "\nOutputs stored successfully for:"
                f"\n\t llm: {llm_str}"
                f"\n\t replicate: {replication} / {stage_config.total_replications}"
                f"\n\t stage: {stage_name}"
                f"\n\t subset: {subset}"
                f"\n\t COMPLETE: {output.complete}"
            )
            
            # self.processor(output).process()
            return None
        
    def retrieves(
        self,
        stage_name: str = None,
        llm_str: str = None,
        replication: int = None,
        subset: str = None
    ):
        if stage_name:
            stage_outputs = self.stage_outputs[stage_name]
            if llm_str and replication:
                output_indx = self._output_index(llm_str, replication, stage_name)
                if isinstance(output_indx, int):
                    output = stage_outputs[output_indx]
                    if subset:
                        return output.retreive(subset)
                    return output
                return None
            return stage_outputs
        return self.stage_outputs
    
    def check_output(
        self,
        sub_path: Path,
        llm_str: str,
        replication: int,
        stage_name: str
    ):
        schema = lazy_import("models.irpd.schemas", f"Stage{stage_name}Schema")
        found_all = self._check_directory(sub_path, llm_str, replication, stage_name, schema)
        if found_all:
            return True
        elif self.batch_request:
            return self._check_batch(sub_path, stage_name, llm_str, schema)
        else:
            return False
    
    def _output_index(self, llm_str: str, replication: int, stage_name: str):
        stage_outputs = self.outputs[stage_name]
        output = next((c for c in stage_outputs if c.llm_str == llm_str and c.replication == replication), None)
        if output:
            return stage_outputs.index(output)
        return None
    
    def _check_directory(
        self,
        sub_path: Path,
        llm_str: str,
        replication: int,
        stage_name: str,
        schema: BaseModel
    ):
        stage_path = sub_path / f"stage_{stage_name}"
        if not check_directories(stage_path):
            log.warning(
                f"OUTPUT: No outputs found for Stage {stage_name}"
                f" in replication {replication} using {llm_str}."
            )
            return False

        meta_json = load_json(sub_path / "_test_meta.json")
        meta = validate_json(meta_json, TestMeta)
        subsets = meta.stages[stage_name].subsets

        stage_output = StageOutput(stage_name, llm_str, replication)
        stage_output.initialize(subsets)

        found_subsets = []

        for subset in subsets:
            subset_path = stage_path / subset
            if not check_directories(subset_path):
                continue

            responses_path = subset_path / "responses"
            responses_parsed = [
                RequestOut(parsed=validate_json(load_json(response), schema))
                for response in responses_path.iterdir()
                if response.name.endswith("response.txt")
            ]
            if responses_parsed:
                stage_output.store(subset, responses_parsed)
                found_subsets.append(subset)
        
        output_indx = self._output_index(llm_str, replication, stage_name)
        if isinstance(output_indx, int):
            self.stage_outputs[stage_name][output_indx] = stage_output
        else:
            self.stage_outputs[stage_name] += [stage_output]

        if stage_output.output_validation(subsets):
            log.info(
                f"OUTPUT: All outputs found for Stage {stage_name}"
                f" in replication {replication} using {llm_str}."
            )
            return True
        elif len(found_subsets):
            log.warning(
                f"OUTPUT: Partial outputs found for Stage {stage_name}"
                f" in replication {replication} using {llm_str}."
                f" Found subsets: {found_subsets}."
            )
        else:
            log.warning(
                f"OUTPUT: No outputs found for Stage {stage_name}"
                f" in replication {replication} using {llm_str}."
            )
            return False
        
    def _check_batch(
        self,
        sub_path: Path,
        stage_name: str,
        llm_str: str,
        schema: BaseModel
    ):
        meta_path = sub_path / "_test_meta.json"
        
        if not meta_path.exists():
            return False
        
        meta_json = load_json(meta_path)
        meta = validate_json(meta_json, TestMeta)
        
        batch_id = meta.stages[stage_name].batch_id
        batch_file = f"stage_{stage_name}_{llm}.jsonl"
        batch_file_path = Path(self.test_path / "_batches" / batch_file)
        
        llm = getattr(LLMModel, llm_str).get_llm_instance(self.llm_config)
        batch = llm.retreive_batch(batch_id, schema, batch_file_path)
        
        if batch:
            log.info(f"OUTPUT: Stage {stage_name} batch complete, storing outputs.")
            
            for response in batch:
                id_split = response.response_id.split("-")
                replication = id_split[0]
                subset = id_split[1]
                
                output_indx = self._output_index(llm_str, replication, stage_name)
                if isinstance(output_indx, int):
                    self.stage_outputs[stage_name][output_indx].store(subset, response.response)
                else:
                    stage_output = StageOutput(stage_name, llm_str, replication)
                    self.stage_outputs[stage_name] += [stage_output.store(subset, response.response)]
        return True
    
    def store(self, subset: str, outputs: Union[RequestOut, List[RequestOut]]):
        if self.outputs[subset]:
            self.outputs[subset].extend(to_list(outputs))
        else:
            self.outputs[subset] = to_list(outputs)
        return None
    
    def retreiveaa(self, subset: str):
        if subset in self.outputs.keys():
            return self.outputs[subset]
        return None