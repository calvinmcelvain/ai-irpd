import logging
from itertools import product
from typing import Optional, List, Union
from pathlib import Path

from logger import clear_logger
from utils import create_directory
from models.irpd.irpd_base import IRPDBase
from models.irpd.test_config import TestConfig
from models.irpd.test_prompts import TestPrompts
from models.irpd.stage import Stage


log = logging.getLogger(__name__)



class SampleSplitting(IRPDBase):
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
        test_paths: Optional[List[str]] = None,
    ):
        super().__init__(
            cases,
            ras,
            treatments,
            stages,
            llms,
            llm_configs,
            output_path,
            prompts_path,
            data_path,
            test_paths
        )
        self._test_type = "sample_splitting"
    
    def _generate_test_paths(self):
        pass
    
    def _generate_test_configs(self):
        pass
    
    def run(
        self,
        max_instances = None,
        config_ids = None,
        print_response = False
    ):
        pass