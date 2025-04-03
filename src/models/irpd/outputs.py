import logging
from typing import Dict, List, Union
from pathlib import Path
from pydantic import BaseModel
from dataclasses import dataclass, field

from utils import lazy_import, check_directories, validate_json, load_json, to_list
from models.irpd.test_configs import TestConfig
from models.request_output import RequestOut
from models.irpd.test_meta import TestMeta
from models.llm_model import LLMModel


log = logging.getLogger(__name__)



@dataclass
class StageOutput:
    stage: str
    llm_str: str
    replication: int
    outputs: Dict[str, List[RequestOut]] = field(default_factory=dict)
    
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


@dataclass
class TestOutput:
    config: TestConfig
    stage_outputs: Dict[str, List[StageOutput]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.complete = False
        self.batch_request = self.config.batches
        self.id = self.config.id
        self.test_path = self.config.test_path
        self.llms = self.config.llms
        self.llm_config = self.config.llm_config
        self.stages = self.config.stages
        self.total_replications = self.config.total_replications
        self.stage_outputs = {stage: [] for stage in self.config.stages}
    
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