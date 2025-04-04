import re
import logging
from typing import List, Optional, Union, Dict
from pathlib import Path
from abc import ABC, abstractmethod

from utils import get_env_var, to_list, str_to_path
from logger import clear_logger
from models.irpd.test_configs import TestConfig
from models.irpd.test_runner import TestRunner
from models.irpd.managers import ConfigManager, OutputManager


log = logging.getLogger(__name__)



class IRPDBase(ABC):
    configs: Dict[str, ConfigManager]
    outputs: Dict[str, OutputManager]
    
    def __init__(
        self, 
        cases: Union[List[str], str],
        ras: Union[List[str], str],
        treatments: Union[List[str], str],
        stages: Union[List[str], str],
        N: int,
        llms: Optional[Union[List[str], str]] = None,
        llm_configs: Optional[Union[List[str], str]] = None,
        output_path: Optional[Union[str, Path]] = None,
        prompts_path: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        test_paths: Optional[List[str]] = None,
        batch: bool = False
    ):
        self.cases = to_list(cases)
        self.ras = to_list(ras)
        self.treatments = to_list(treatments)
        self.stages = to_list(stages)
        self.llms = to_list(llms)
        self.llm_configs = to_list(llm_configs)
        self.test_paths = to_list(test_paths or [])
        self.batch_request = batch
        
        assert N >= 1, "`N` must be greater than 0."
        self.replications = N

        self.output_path = str_to_path(output_path or get_env_var("OUTPUT_PATH"))
        self.prompts_path = str_to_path(prompts_path or get_env_var("PROMPTS_PATH"))
        self.data_path = str_to_path(data_path or get_env_var("DATA_PATH"))
    
    def _validate_test_paths(self):
        test_paths = [Path(path) for path in self.test_paths]
        if not len(self.test_paths) == len(self._prod):
            log.error(
                "test_paths must be the same length as the number of test configs."
            )
            raise ValueError(
                "test_paths must be the same length as the number of test configs."
            )
        return test_paths

    @staticmethod
    def _get_max_test_number(directory: Path, prefix: str = "test_"):
        pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
        matches = (
            match.group(1) for p in directory.iterdir()
            if (match := pattern.match(p.name))
        )
        return max(map(int, matches), default=0)
    
    def _get_test_configs(self, config_ids: Union[str, List[str]]):
        if config_ids:
            config_ids = to_list(config_ids)
            return [
                self.configs[config_id] for config_id in config_ids
                if config_id in self.configs.test_configs.keys()
            ]
        else:
            return self.configs.values()

    def add_configs(self, configs: Union[TestConfig, List[TestConfig]]):
        configs = to_list(configs)
        for config in configs:
            if not isinstance(config, TestConfig):
                log.error(
                    f"Test config {config} was not a TestConfig instance."
                    " Did not add."
                )
                continue
            if config.test_type not in self._test_type:
                log.error(
                    f"Test config {config.test_id} was not correct test type."
                    " Did not add."
                )
                continue
            self.configs.add(config)

    @abstractmethod
    def _generate_test_paths(self):
        pass
    
    @abstractmethod
    def _generate_configs(self):
        pass
    
    async def run(
        self,
        max_instances: Optional[int] = None,
        config_ids: Union[str, List[str]] = None,
        print_response: bool = False
    ):
        clear_logger(app=False)
        test_configs = self._get_test_configs(config_ids=config_ids)
        
        for config in test_configs:
            test_runner = TestRunner(config, max_instances, print_response)
            await test_runner.run()