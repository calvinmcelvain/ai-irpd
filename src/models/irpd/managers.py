from pathlib import Path
from itertools import product
from typing import Dict, List, Optional

from models.irpd.test_configs import TestConfig, SubConfig, StageConfig
from models.irpd.outputs import TestOutput, SubOutput, StageOutput
from models.llm_model import LLMModel



class ConfigManager:
    def __init__(self, test_configs: Dict[str, TestConfig]):
        self.test_configs = test_configs
        self.sub_configs = self._generate_sub_configs(self.test_configs)
        self.stage_configs = self._generate_stage_configs(self.sub_configs)
        
    def _generate_sub_configs(self, configs: Dict[str, TestConfig]):
        sub_configs = {}
        for key, value in configs:
            sub_configs[key] = [SubConfig(
                **vars(value),
                sub_path=self._generate_subpath(n, llm_str),
                llm=llm_str,
                llm_instance=self._generate_llm_instance(llm_str),
                replication=n
            ) 
            for llm_str, n in product(value.llms, range(1, value.total_replications + 1))
            ]
        return sub_configs
    
    def _generate_stage_configs(self, configs: Dict[str, List[SubConfig]]):
        stage_configs = {}
        for key, _, value in configs:
            stage_configs[key] = [
                StageConfig(**vars(value),stage_name=stage)
                for stage in value.stages
            ]
        return stage_configs
    
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
        config_id: Optional[str],
        llm_str: Optional[str],
        N: Optional[int],
        stage: Optional[str]
    ):
        if config_id:
            stage_configs = self.stage_configs[config_id]
            if llm_str:
                llm_configs = [config for config in stage_configs if config.llm == llm_str]
                if isinstance(N, int):
                    replicate_configs = [config for config in llm_configs if config.replication == N]
                    if stage:
                        return next((config for config in replicate_configs if config.stage_name == stage))
                    return replicate_configs
                return llm_configs
            return stage_configs
        return None
    
    def add(self, config: TestConfig):
        self.test_configs[config.id] = config
        self.sub_configs.update(self._generate_sub_configs({config.id: config}))
        self.stage_configs.update(self._generate_stage_configs({config.id: self.sub_configs[config.id]}))
        


class OutputManager:
    def __init__(self):
        pass
    
    def store(
        self,
        stage_name: str,
        llm_str: str,
        replication: int,
        subset: str,
        outputs: Union[RequestOut, List[RequestOut]]
    ):
        output_idx = self._output_index(llm_str, replication, stage_name)
        if isinstance(output_idx, int):
            output = self.stage_outputs[stage_name][output_idx]
            output.store(subset, outputs)
        else:
            stage_output = StageOutput(stage_name, llm_str, replication)
            stage_output.store(subset, outputs)
            self.stage_outputs[stage_name].append(stage_output)
        return None
    
    def retrieve(
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
    
    def retreive(self, subset: str):
        if subset in self.outputs.keys():
            return self.outputs[subset]
        return None