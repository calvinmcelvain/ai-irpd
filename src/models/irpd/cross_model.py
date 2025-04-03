import logging
from time import sleep
from itertools import product
from typing import Optional, List, Union
from pathlib import Path

from logger import clear_logger
from utils import create_directory
from models.irpd.irpd_base import IRPDBase
from models.irpd.outputs import TestOutput
from models.irpd.test_configs import TestConfig
from models.irpd.test_prompts import TestPrompts
from models.irpd.managers import ConfigManager
from models.irpd.stage import Stage


log = logging.getLogger(__name__)



class CrossModel(IRPDBase):
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
        super().__init__(
            cases,
            ras,
            treatments,
            stages,
            N,
            llms,
            llm_configs,
            output_path,
            prompts_path,
            data_path,
            test_paths,
            batch
        )
        self.test_type = "cross_model"
        self._prod = list(product(
            self.llm_configs, self.cases, self.ras, self.treatments
        ))
        
        self.test_paths = self._generate_test_paths()
        self._generate_configs()
    
    def _generate_test_paths(self):
        if self.test_paths:
            return self._validate_test_paths()
        test_dir = self.output_path / self.test_type
        current_test = self._get_max_test_number(test_dir)
        test_paths = [test_dir / f"test_{i + 1 + current_test}" for i in range(len(self._prod))]
        return test_paths
    
    def _generate_configs(self):
        test_configs = {}
        for idx, prod in enumerate(self._prod):
            llm_config, case, ra, treatment = prod
            config = TestConfig(
                case=case,
                ra=ra,
                treatment=treatment,
                llms=self.llms,
                llm_config=llm_config,
                test_type=self.test_type,
                test_path=self.test_paths[idx],
                stages=self.stages
            )
            test_configs[config.id] = config
        self.configs = ConfigManager(test_configs)
        return None
    