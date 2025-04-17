"""
IRPDBase module.

Contains the IRPDBase model that specifies the `run` method and validation 
methods.
"""
import re
import logging
from typing import List, Optional, Union, Dict, Tuple
from pathlib import Path
from abc import ABC, abstractmethod

from helpers.utils import get_env_var, to_list, load_config
from core.test_runner import TestRunner
from core.output_manager import OutputManager
from _types.irpd_config import IRPDConfig


CONFIGS: Dict = load_config("irpd.json")
PARAMETERS: Dict = CONFIGS["parameters"]


log = logging.getLogger("app")



class IRPDBase(ABC):
    configs: Dict[str, IRPDConfig] = {}
    outputs: Dict[str, OutputManager] = {}
    
    def __init__(
        self, 
        cases: Union[List[str], str],
        ras: Union[List[str], str],
        treatments: Union[List[str], str],
        stages: Union[List[str], str],
        N: int,
        context: Optional[Tuple[int, int]],
        llms: Union[List[str], str],
        llm_configs: Union[List[str], str],
        max_instances: Optional[int],
        max_summaries: Optional[int],
        output_path: Optional[Union[str, Path]],
        prompts_path: Optional[Union[str, Path]],
        data_path: Optional[Union[str, Path]],
        test_paths: Optional[Union[List[Union[str, Path]], Union[str, Path]]],
        batch: bool,
        test_type: str
    ):
        self.test_type = test_type
        self.cases = to_list(cases)
        self.ras = to_list(ras)
        self.treatments = to_list(treatments)
        self.stages = to_list(stages)
        self.llms = to_list(llms)
        self.llm_configs = to_list(llm_configs)
        self.test_paths = to_list(test_paths or [])
        self.batch_request = batch
        self.context = context
        
        self._validate_test_parameters()
        
        if max_instances:
            assert max_instances >= 1, "`max_instances` must be greater than 0."
        if max_summaries:
            assert max_summaries >= 1, "`max_summaries` must be greater than 0."
        self.max_instances = max_instances
        self.max_summaries = max_summaries
        
        assert N >= 1, "`N` must be greater than 0."
        self.replications = N

        # If paths not specified, using env variable.
        self.output_path = Path(output_path or get_env_var("OUTPUT_PATH"))
        self.prompts_path = Path(prompts_path or get_env_var("PROMPTS_PATH"))
        self.data_path = Path(data_path or get_env_var("DATA_PATH"))
    
    def _validate_test_parameters(self) -> None:
        """
        Validates test parameters from `irpd.json` configs file & reindexes them.
        """
        for param, valid_params in PARAMETERS.items():
            self_param = getattr(self, param)
            if not set(self_param).issubset(valid_params):
                raise ValueError(f"`{param}` must be from {valid_params}.")
            order = {key: index for index, key in enumerate(valid_params)}
            setattr(self, param, sorted(self_param, key=order.get))
        return None
                    
    def _validate_test_paths(self) -> List[Path]:
        """
        Ensures all test paths are Path objects and ensures that the number of
        test paths specified are the same length as the calculated number of
        tests.
        """
        test_paths = [Path(path) for path in self.test_paths]
        if not len(self.test_paths) == len(self._prod):
            log.error(
                "`test_paths` must be the same length as the number of test configs."
            )
            raise ValueError(
                "`test_paths`must be the same length as the number of test configs."
            )
        return test_paths

    @staticmethod
    def _get_max_test_number(directory: Path, prefix: str = "test_") -> int:
        """
        Gets the maximum test number for a given test directory.
        """
        pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
        matches = (
            match.group(1) for p in directory.iterdir()
            if (match := pattern.match(p.name))
        )
        return max(map(int, matches), default=0)
    
    def _get_configs(
        self, config_ids: Union[str, List[str]]
    ) -> Dict[str, IRPDConfig]:
        """
        Returns dictionary of IRPD configs based on a list of config ids. 
        Otherwise returns configs attrb.
        """
        if config_ids:
            config_ids = to_list(config_ids)
            return {k: self.configs[k] for k in config_ids if k in self.configs}
        else:
            return self.configs

    @abstractmethod
    def _generate_test_paths(self) -> None:
        """
        Generates test paths for each IRPD config. Different for each test type.
        """
        pass
    
    @abstractmethod
    def _generate_configs(self) -> None:
        """
        Generates the IRPD configs based on instance args. Difference for each
        test type.
        """
        pass
    
    def run(
        self,
        config_ids: Union[str, List[str]] = None,
        print_response: bool = False
    ) -> None:
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
        irpd_configs = self._get_configs(config_ids=config_ids)
        
        for config_id, config in irpd_configs.items():
            test_runner = TestRunner(config, print_response)
            self.outputs[config_id] = test_runner.run()
        return None