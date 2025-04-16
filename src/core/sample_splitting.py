"""
The sample splitting test module.

Contains the SampleSplitting model.
"""
import logging
from typing import Optional, List, Union
from pathlib import Path

from core.base import IRPDBase
from _types.irpd_config import IRPDConfig


log = logging.getLogger(__name__)



class SampleSplitting(IRPDBase):
    def __init__(
        self, 
        cases: Union[List[str], str],
        ras: Union[List[str], str],
        treatments: Union[List[str], str],
        stages: Union[List[str], str],
        N: int = 1,
        llms: Optional[Union[List[str], str]] = None,
        llm_configs: Optional[Union[List[str], str]] = None,
        max_instances: Optional[int] = None,
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
            max_instances,
            output_path,
            prompts_path,
            data_path,
            test_paths,
            batch
        )
        self._test_type = "sample_splitting"
    
    def _generate_test_paths(self):
        pass
    
    def _generate_test_configs(self):
        pass