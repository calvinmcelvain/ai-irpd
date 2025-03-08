import re
import logging
from typing import List, Dict, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod

from utils import get_env_var, to_list, load_config, str_to_path
from models.llm_model import LLMModel
from models.irpd.test_output import TestOutput
from models.irpd.test_config import TestConfig


CONFIGS = load_config("irpd_configs.yml")
DEFAULTS = CONFIGS["defaults"]
VALID_VALUES = CONFIGS["valid_values"]

log = logging.getLogger(__name__)



class IRPDBase(ABC):    
    output: Dict[str, List[TestOutput]]
    configs: Dict[str, TestConfig]
    
    def __init__(
        self, 
        case: str,
        ras: Optional[List[str]],
        treatments: Optional[List[str]],
        stages: Optional[List[str]] = None,
        llms: Optional[List[str]] = None,
        llm_configs: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None,
        prompts_path: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        test_paths: Optional[List[str]] = None
    ):
        self.case = case
        self.ras = ras or []
        self.treatments = treatments or []
        self.stages = stages or []
        self.llms = llms or []
        self.llm_configs = llm_configs or []
        self.test_paths = test_paths or []

        self._validate_values()

        self.output_path = str_to_path(output_path or get_env_var("OUTPUT_PATH"))
        self.prompts_path = str_to_path(prompts_path or get_env_var("PROMPTS_PATH"))
        self.data_path = str_to_path(data_path or get_env_var("DATA_PATH"))

    def _validate_values(self):
        attributes = ["case", "ras", "treatments", "stages", "llms", "llm_configs"]
        for attr in attributes:
            value = getattr(self, attr)
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
                raise ValueError(
                    f"All provided `{attr}` values are invalid: {value}. "
                    f"Allowed values: {valid_values}"
                )
            if invalid_items:
                log.warning(
                    f"Some `{attr}` values were ignored as invalid: {invalid_items}. "
                    f"Allowed values: {valid_values}"
                )
            setattr(self, attr, valid_items)
        self.case = self.case[0] if isinstance(self.case, list) else self.case

    def _ensure_strings(self, attr: str, values: List[str]):
        if not all(isinstance(item, str) for item in values):
            log.error(f"Argument `{attr}` must contain only string values.")
            raise TypeError(f"Argument `{attr}` must contain only string values.")

    def _filter_valid_items(self, values: List[str], valid_values: List[str]):
        valid_items = [item for item in values if item in valid_values]
        invalid_items = [item for item in values if item not in valid_values]
        return valid_items, invalid_items
    
    def _generate_llm_instance(
        self,
        llm: str,
        config: str,
        print_response: bool = False
    ):
        return getattr(LLMModel, llm).get_llm_instance(
            config=config, print_response=print_response
        )
    
    @staticmethod
    def _get_max_test_number(directory: Path, prefix: str = "test_"):
        pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
        return max(
            map(int, (match.group(1) for p in directory.iterdir() if (match := pattern.match(p.name)))),
            default=0
        )
    
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
        config_ids = to_list(config_ids)
        if config_ids:
            self._test_configs = {k: self.configs[k] for k in config_ids if k in self.configs}
        else:
            self._test_configs = self.configs