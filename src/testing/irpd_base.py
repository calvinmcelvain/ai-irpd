import re
import logging
from typing import List, Dict
from pathlib import Path
from abc import ABC, abstractmethod
from utils import get_env_var
from llms import *
from test_config import TestConfig
from output_manager import OutputManager

log = logging.getLogger("app.testing.irpd_base")


class IRPDBase(ABC):
    _VALID_STAGES = ['0', '1', '1r', '1c', '2', '3']
    _VALID_CASES = ["uni", "uniresp", "switch", "first", "uni_switch"]
    _VALID_RAS = ["ra1", "ra2", "both", "exp"]
    _VALID_TREATMENTS = ["imperfect", "perfect", "merged"]
    _VALID_LLMS = LLMModel._member_names_
    _VALID_LLM_CONFIGS = ["base", "res1", "res2", "res3"]
    
    OUTPUTS: OutputManager = OutputManager()
    TEST_CONFIGS: Dict[str, TestConfig]

    def __init__(
        self, 
        case: str,
        ras: List[str],
        treatments: List[str],
        stages: List[str],
        llms: List[str] = ["GPT_4O_1120"],
        llm_configs: List[str] = ["base"],
        project_path: str = None,
        new_test: bool = True
    ):
        self.case = self._validate_arg(
            [case], self._VALID_CASES, "cases")[0]
        self.ras = self._validate_arg(
            ras, self._VALID_RAS, "ras")
        self.treatments = self._validate_arg(
            treatments, self._VALID_TREATMENTS, "treatments")
        self.stages = self._validate_arg(
            stages, self._VALID_STAGES, "stages")
        self.llms = self._validate_arg(
            llms, self._VALID_LLMS, "llms")
        self.llm_configs = self._validate_arg(
            llm_configs, self._VALID_LLM_CONFIGS, "llm_configs")
        self.project_path = Path(
            project_path if project_path else get_env_var("PROJECT_DIRECTORY")
        )
        self.output_path = self.project_path / "output"
        self.new_test = new_test

    def _validate_arg(self, arg: list[str], valid_values: list[str], name: str):
        if not isinstance(arg, list) or not all(isinstance(item, str) for item in arg):
            arg = [arg]
            setattr(self, name, [arg])

        valid_set = set(valid_values)
        index_map = {value: i for i, value in enumerate(valid_values)} 

        valid_items, invalid_items = [], []
        for item in arg:
            (valid_items if item in valid_set else invalid_items).append(item)

        if not valid_items:
            raise ValueError(
                f"All provided `{name}` values are invalid. No valid items remain. "
                f"Allowed values: {valid_values}", 
            )
        
        if invalid_items:
            log.warning(
                f"Some `{name}` values are invalid and were ignored: {invalid_items}. "
                f"Allowed values: {valid_values}"
            )

        return sorted(valid_items, key=lambda x: index_map[x])
    
    def _generate_model_instance(self, llm: str, config: str):
        return getattr(LLMModel, llm).get_model_instance(
                config=config, print_response=self.print_response
        )
    
    @staticmethod
    def _get_max_test_number(directory: Path, prefix: str = "test_"):
        pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
        return max(
            map(int, (match.group(1) for p in directory.iterdir() if (match := pattern.match(p.name)))),
            default=0
        )

    @abstractmethod
    def _generate_test_path(self):
        pass
    
    def _generate_sub_path(test_config: TestConfig, n: int):
        pass
    
    @abstractmethod
    def _generate_test_configs(self):
        pass
    
    @abstractmethod
    def run(self, max_instances: int = None, threshold: float = 0.5):
        pass