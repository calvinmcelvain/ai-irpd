"""
IRPDBase module.

Contains the IRPDBase model that specifies the `run` method and validation 
methods.
"""
import re
import logging
from typing import List, Optional, Union, Dict
from pathlib import Path
from abc import ABC, abstractmethod

from helpers.utils import get_env_var, to_list
from logger import LoggerManager
from _types.irpd_config import IRPDConfig
from core.test_runner import TestRunner


log = logging.getLogger(__name__)



class IRPDBase(ABC):
    def __init__(
        self, 
        cases: Union[List[str], str],
        ras: Union[List[str], str],
        treatments: Union[List[str], str],
        stages: Union[List[str], str],
        N: int,
        llms: Optional[Union[List[str], str]] = None,
        llm_configs: Optional[Union[List[str], str]] = None,
        max_instances: Optional[int] = None,
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
        
        if max_instances:
            assert max_instances >= 1, "`max_instances` must be greater than 0."
        self.max_instances = max_instances
        
        assert N >= 1, "`N` must be greater than 0."
        self.replications = N

        # If paths not specified, using env variable.
        self.output_path = Path(output_path or get_env_var("OUTPUT_PATH"))
        self.prompts_path = Path(prompts_path or get_env_var("PROMPTS_PATH"))
        self.data_path = Path(data_path or get_env_var("DATA_PATH"))
        
        # Initializing configs and outputs attrbs.
        self.configs = {}
        self.outputs = {}
    
    def _validate_test_paths(self):
        """
        Ensures all test paths are Path objects and ensures that the number of
        test paths specified are the same length as the calculated number of
        tests.
        """
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
        """
        Gets the maximum test number for a given test directory.
        """
        pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
        matches = (
            match.group(1) for p in directory.iterdir()
            if (match := pattern.match(p.name))
        )
        return max(map(int, matches), default=0)
    
    def _get_test_configs(self, config_ids: Union[str, List[str]]):
        """
        Returns dictionary of IRPD configs based on a list of config ids. 
        Otherwise returns configs attrb.
        """
        if config_ids:
            config_ids = to_list(config_ids)
            return {k: self.configs[k] for k in config_ids if k in self.configs}
        else:
            return self.configs

    def add_configs(self, configs: Union[IRPDConfig, List[IRPDConfig]]):
        """
        Method to add IRPD configs to configs attrb.
        """
        configs = to_list(configs)
        for config in configs:
            if not isinstance(config, IRPDConfig):
                log.error(
                    f"Test config {config} was not a IRPDConfig instance."
                    " Did not add."
                )
                continue
            if config.test_type not in self._test_type:
                log.error(
                    f"Test config {config.test_id} was not correct test type."
                    " Did not add."
                )
                continue
            self.configs[config.id] = config

    @abstractmethod
    def _generate_test_paths(self):
        """
        Generates test paths for each IRPD config. Different for each test type.
        """
        pass
    
    @abstractmethod
    def _generate_configs(self):
        """
        Generates the IRPD configs based on instance args. Difference for each
        test type.
        """
        pass
    
    def run(
        self,
        config_ids: Union[str, List[str]] = None,
        print_response: bool = False
    ):
        """
        Runs IRPD based on the defined test configs.

        Args:
            config_ids (Union[str, List[str]], optional): If specified, will 
            only run the config ids defined. Otherwise runs all test configs. 
            Defaults to None.
            print_response (bool, optional): If True, prints the LLM request for
            each chat completion request. If batch, this arg. is null. Defaults 
            to False.
        """
        log_manager = LoggerManager()
        log_manager.clear_logs()
        
        irpd_configs: Dict[str, IRPDConfig] = self._get_test_configs(config_ids=config_ids)
        
        for config_id, config in irpd_configs.items():
            test_runner = TestRunner(config, print_response)
            self.outputs[config_id] = test_runner.run()
        return None