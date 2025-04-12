"""
Intra-model IRPD test module.

Contains the IntraModel model.
"""
import logging
from itertools import product
from typing import Optional, List, Union
from pathlib import Path

from core.models.base import IRPDBase
from core.output_manager import OutputManager
from types.irpd_config import IRPDConfig


log = logging.getLogger(__name__)



class IntraModel(IRPDBase):
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
        self.test_type = "cross_model"
        
        # For intra-model tests, the number of tests is the total combinations
        # of LLMs, LLM configs, cases, RAs, and treatments.
        self._prod = list(product(
            self.llms, self.llm_configs, self.cases, self.ras, self.treatments
        ))
        
        self.test_paths = self._generate_test_paths()
        self._generate_configs()
    
    def _generate_test_paths(self):
        if self.test_paths:
            return self._validate_test_paths()
        
        # Tests are in directorys: .../outputs/intra_model/
        test_dir = self.output_path / self.test_type
        current_test = self._get_max_test_number(test_dir)
        test_paths = [test_dir / f"test_{i + 1 + current_test}" for i in range(len(self._prod))]
        return test_paths
    
    def _generate_configs(self):
        for idx, prod in enumerate(self._prod):
            llm, llm_config, case, ra, treatment = prod
            config = IRPDConfig(
                case=case,
                ra=ra,
                treatment=treatment,
                llms=llm,
                llm_config=llm_config,
                max_instances=self.max_instances,
                test_type=self.test_type,
                data_path=self.data_path,
                prompts_path=self.prompts_path,
                test_path=self.test_paths[idx],
                batches=self.batch_request,
                stages=self.stages
            )
            self.configs[config.id] = config
            self.outputs[config.id] = OutputManager(config)
        return None
