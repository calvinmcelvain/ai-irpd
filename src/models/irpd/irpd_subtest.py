"""
The subtest IRPD test module.

Contains the Subtest test model.
"""
import logging
from itertools import product
from typing import Union, Optional, List
from pathlib import Path

from utils import to_list
from models.irpd.irpd_base import IRPDBase
from models.irpd.test_config import TestConfig
from models.irpd.output_manager import OutputManager


log = logging.getLogger(__name__)



class Subtest(IRPDBase):
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
        self.test_type = "subtest"
        
        # The total number of tests is the total combinations of LLMs, LLM
        # configs, cases, RAs, and treatments.
        self._prod = list(product(
            self.llms, self.llm_configs, self.cases, self.ras, self.treatments
        ))
        
        assert self.replications == 1, "For test type 'test' or 'subtest', replications `N` must be equal to 1"
        
        self.test_paths = self._generate_test_paths()
        self._generate_configs()
    
    def _generate_test_paths(self):
        if self.test_paths:
            return self._validate_test_paths()
        test_dir = self.output_path / "subtests"
        current_test = self._get_max_test_number(test_dir, "")
        test_paths = [test_dir / f"{i + 1 + current_test}" for i in range(len(self._prod))]
        return test_paths
    
    def _generate_configs(self):
        for idx, prod in enumerate(self._prod):
            llm, llm_config, case, ra, treatment = prod
            config = TestConfig(
                case=case,
                ra=ra,
                treatment=treatment,
                llms=to_list(llm),
                llm_config=llm_config,
                test_type=self.test_type,
                max_instances=self.max_instances,
                test_path=self.test_paths[idx],
                data_path=self.data_path,
                prompts_path=self.prompts_path,
                stages=self.stages,
                batches=self.batch_request,
                total_replications=1
            )
            self.configs[config.id] = config
            self.outputs[config.id] = OutputManager(config)
        return None
