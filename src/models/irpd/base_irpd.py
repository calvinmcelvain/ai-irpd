import re
import logging
from typing import List, Dict, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod

from utils import (
    get_env_var, to_list, load_config, str_to_path, validate_json_string,
    file_to_string, lazy_import
)
from models.llm_model import LLMModel
from models.request_output import RequestOut
from models.irpd.test_output import TestOutput
from models.irpd.test_config import TestConfig
from models.irpd.stage_output import StageOutput


CONFIGS = load_config("irpd_configs.yml")
DEFAULTS = CONFIGS["defaults"]
VALID_VALUES = CONFIGS["valid_values"]


log = logging.getLogger(__name__)



class IRPDBase(ABC):
    def __init__(
        self, 
        cases: Union[List[str], str],
        ras: Union[List[str], str],
        treatments: Union[List[str], str],
        stages: Union[List[str], str],
        llms: Optional[Union[List[str], str]] = None,
        llm_configs: Optional[Union[List[str], str]] = None,
        output_path: Optional[Union[str, Path]] = None,
        prompts_path: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        test_paths: Optional[List[str]] = None
    ):
        self.cases = to_list(cases)
        self.ras = ras or []
        self.treatments = treatments or []
        self.stages = stages or []
        self.llms = llms or []
        self.llm_configs = llm_configs or []
        self.test_paths = test_paths or []
        
        self.output = {}
        self.configs = {}

        self._validate_values()

        self.output_path = str_to_path(output_path or get_env_var("OUTPUT_PATH"))
        self.prompts_path = str_to_path(prompts_path or get_env_var("PROMPTS_PATH"))
        self.data_path = str_to_path(data_path or get_env_var("DATA_PATH"))

    def _validate_values(self):
        attributes = ["cases", "ras", "treatments", "stages", "llms", "llm_configs"]
        for attr in attributes:
            value = getattr(self, attr)
            if attr == "cases":
                vals = []
                for c in value:
                    vals.extend(c.split("_"))
                value = list(set(vals))
            default_value = to_list(DEFAULTS.get(attr, ""))
            valid_values = VALID_VALUES.get(attr, [])

            if not valid_values:
                log.warning(f"No valid values found for `{attr}` in irpd configs.")
            if not value:
                setattr(self, attr, default_value)
                continue

            value = to_list(value)
            self._ensure_strings(attr, value)
            valid_items, invalid_items = self._filter_valid_items(value, valid_values)

            if not valid_items:
                log.error(
                    f"All provided `{attr}` values are invalid: {value}. "
                    f"Allowed values: {valid_values}"
                )
                raise ValueError(
                    f"All provided `{attr}` values are invalid: {value}. "
                    f"Allowed values: {valid_values}"
                )
            if attr != "cases":
                if invalid_items:
                    log.warning(
                        f"Some `{attr}` values were ignored as invalid: {invalid_items}. "
                        f"Allowed values: {valid_values}"
                    )
                setattr(self, attr, valid_items)

    def _ensure_strings(self, attr: str, values: List[str]):
        if not all(isinstance(item, str) for item in values):
            log.error(f"Argument `{attr}` must contain only string values.")
            raise TypeError(f"Argument `{attr}` must contain only string values.")

    def _filter_valid_items(self, values: List[str], valid_values: List[str]):
        valid_items = [item for item in values if item in valid_values]
        invalid_items = [item for item in values if item not in valid_values]
        return valid_items, invalid_items
    
    def _validate_test_paths(self):
        test_paths = [Path(path) for path in to_list(self.test_paths)]
        if not len(self.test_paths) == len(self._prod):
            log.error(
                "test_paths must be the same length as the number of test configs."
            )
            raise ValueError(
                "test_paths must be the same length as the number of test configs."
            )
        return test_paths
    
    def _generate_llm_instance(
        self,
        llm: str,
        config: str,
        print_response: bool = False
    ):
        return getattr(LLMModel, llm).get_llm_instance(
            config=config, print_response=print_response
        )
    
    def _output_indx(
        self,
        id: str,
        llm: str,
        replication: int
    ):
        test = self.output[id]
        test_out = next((c for c in test if c.llm == llm and c.replication == replication), None)
        if test_out:
            return test.index(test_out)
        return None
    
    def _update_output(
        self,
        config: TestConfig,
        llm: str,
        replication: int,
        sub_path: Path
    ):
        log.info("OUTPUT: Checking for output.")
        exist_stgs = [s for s in config.stages if (sub_path / f"stage_{s}").exists()]
        if exist_stgs:
            test_out = {}
            for s in exist_stgs:
                log.info(f"OUTPUT: Stage {s} found.")
                stage_out = {}
                stage_path = sub_path / f"stage_{s}"
                subsets = [s for s in stage_path.iterdir() if s.is_dir()]
                for k in subsets:
                    stage_out[k.name] = []
                    schema = lazy_import("models.schemas", f"Stage{s}Schema")
                    if s in {"2", "3"}:
                        r_dir = k / "responses"
                    else:
                        r_dir = k
                    for r in r_dir.iterdir():
                        if r.name.endswith("response.txt"):
                            parsed = validate_json_string(r, schema)
                            stage_out[k.name].append(RequestOut(
                                text=file_to_string(r),
                                parsed=parsed
                            ))
                test_out[s] = StageOutput(stage=s, outputs=stage_out)
            self.output[config.id].append(TestOutput(
                id=config.id,
                llm=llm,
                replication=replication,
                stage_outputs=test_out
            ))
        else:
            log.info("OUTPUT: No outputs not found.")
        return None
    
    def _get_context(
        self,
        config: TestConfig,
        llm: str,
        replication: int
    ):
        test_idx = self._output_indx(id=config.id, llm=llm, replication=replication)
        if test_idx:
            return self.output[config.id][test_idx]
        else:
            test_out = TestOutput(
                id=config.id,
                llm=llm,
                replication=replication
            )
            self.output[config.id].append(test_out)
        return None

    @staticmethod
    def _get_max_test_number(directory: Path, prefix: str = "test_"):
        pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
        return max(
            map(int, (match.group(1) for p in directory.iterdir() if (match := pattern.match(p.name)))),
            default=0
        )
    
    def _get_test_configs(self, config_ids: Union[str, List[str]]):
        config_ids = to_list(config_ids)
        if config_ids:
            return {k: self.configs[k] for k in config_ids if k in self.configs}
        else:
            return self.configs
    
    def remove_configs(self, config_ids: Union[str, List[str]]):
        config_ids = to_list(config_ids)
        for id in config_ids:
            del self.configs[id]
        return None

    def add_configs(self, configs: Union[TestConfig, List[TestConfig]]):
        configs = to_list(configs)
        for config in configs:
            if not isinstance(config, TestConfig):
                log.error(f"Test config {config} was not a TestConfig instance. Did not add.")
                continue
            if config.test_type not in self._test_type:
                log.error(f"Test config {config.test_id} was not correct test type. Did not add.")
                continue
            self.configs[config.test_id] = config

    @abstractmethod
    def _generate_test_paths(self):
        pass
    
    @abstractmethod
    def _generate_configs(self):
        pass
    
    @abstractmethod
    def run(
        self,
        max_instances: Optional[int] = None,
        config_ids: Union[str, List[str]] = None,
        print_response: bool = False
    ):
        pass