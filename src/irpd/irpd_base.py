import re
import logging
from typing import List
from pathlib import Path
from abc import ABC, abstractmethod
from utils import get_env_var, str_to_list, load_config
from llms import *
from irpd.test_config import TestConfig
from irpd.output_manager import OutputManager

log = logging.getLogger(__name__)

CONFIGS = load_config()
DEFAULTS = CONFIGS["defaults"]
VALID_VALUES = CONFIGS["valid_values"]


class IRPDBase(ABC):    
    OUTPUTS: OutputManager = OutputManager()

    def __init__(
        self, 
        case: str,
        ras: List[str],
        treatments: List[str],
        stages: List[str] = None,
        llms: List[str] = None,
        llm_configs: List[str] = None,
        project_path: str | Path = None,
        new_test: bool = True,
        test_paths: List = None
    ):
        self.case = case
        self.ras = ras
        self.treatments = treatments
        self.stages = stages
        self.llms = llms
        self.llm_configs = llm_configs
        self.project_path = project_path
        self.new_test = new_test
        self.test_paths = test_paths
        self._validate_values()

        if not project_path:
            self.project_path = Path(get_env_var("PROJECT_DIRECTORY"))
        self.output_path = self.project_path / "output"
        self.configs = {}

    def _validate_values(self):
        attributes = ["case", "ras", "treatments", "stages", "llms", "llm_configs"]
        for attr in attributes:
            value = getattr(self, attr)
            if value is None:
                setattr(self, attr, str_to_list(DEFAULTS[attr]))
            else:
                valid_values = VALID_VALUES[attr]
                value = str_to_list(value)
                if not all(isinstance(item, str) for item in value):
                    log.error(f"Argument {attr} must have only string value(s)")
                    raise ValueError(f"Argument {attr} must have only string value(s)")

                index_map = {v: i for i, v in enumerate(valid_values)}

                valid_items, invalid_items = [], []
                for item in value:
                    (valid_items if item in valid_values else invalid_items).append(item)

                if not valid_items:
                    log.error(
                        f"All provided `{attr}` values are invalid. No valid items remain."
                        f" Allowed values: {valid_values}"
                    )
                    raise ValueError(f"All provided `{attr}` values are invalid. No valid items remain.")

                if invalid_items:
                    log.warning(
                        f"Some `{attr}` values are invalid and were ignored: {invalid_items}. "
                        f"Allowed values: {valid_values}"
                    )
                setattr(self, attr, sorted(valid_items, key=lambda x: index_map[x]))
        self.case = self.case[0]    # Needs to be string, not list
    
    def _generate_model_instance(self, llm: str, config: str):
        return getattr(LLMModel, llm).get_model_instance(config=config)
    
    @staticmethod
    def _get_max_test_number(directory: Path, prefix: str = "test_"):
        pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
        return max(
            map(int, (match.group(1) for p in directory.iterdir() if (match := pattern.match(p.name)))),
            default=0
        )
    
    def remove_configs(self, config_ids: str | List[str]):
        config_ids = str_to_list(config_ids)
        for id in config_ids:
            del self.configs[id]
        return None

    def add_configs(self, configs: TestConfig | List[TestConfig]):
        configs = str_to_list(configs)
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
        max_instances: int = None,
        threshold: float = 0.5,
        config_ids: str | List[str] = None
    ):
        config_ids = str_to_list(config_ids)
        if config_ids:
            self._test_configs = {k: self.configs[k] for k in config_ids if k in self.configs}
        else:
            self._test_configs = self.configs