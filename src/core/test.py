"""
The IRPD base test module.

Contains the Test model.
"""
import logging
from itertools import product
from typing import Optional, List, Union, Tuple
from pathlib import Path

from helpers.utils import to_list
from core.base import IRPDBase
from _types.irpd_config import IRPDConfig


log = logging.getLogger("app")



class Test(IRPDBase):
    def __init__(
        self, 
        cases: Union[List[str], str],
        ras: Union[List[str], str],
        treatments: Union[List[str], str],
        stages: Union[List[str], str],
        N: int = 1,
        context: Optional[Tuple[int, int]] = None,
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
            context,
            llms,
            llm_configs,
            max_instances,
            output_path,
            prompts_path,
            data_path,
            test_paths,
            batch
        )
        self.test_type = "test"
        
        # The total number of tests is the number of combinations of LLMs, LLM
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
        test_paths = []
        
        # Test paths are dependent on the case: .../outputs/base_tests/{case}/
        for case in self.cases:
            test_dir = self.output_path / "base_tests" / case
            current_test = self._get_max_test_number(test_dir)
            length = int(len(self._prod) / len(self.cases))
            paths = [test_dir / f"test_{i + 1 + current_test}" for i in range(length)]
            test_paths.extend(paths)
        return test_paths
    
    def _generate_configs(self):
        for idx, prod in enumerate(self._prod):
            llm, llm_config, case, ra, treatment = prod
            config = IRPDConfig(
                case=case,
                ra=ra,
                treatment=treatment,
                llms=to_list(llm),
                llm_config=llm_config,
                test_type=self.test_type,
                test_path=self.test_paths[idx].as_posix(),
                data_path=self.data_path.as_posix(),
                prompts_path=self.prompts_path.as_posix(),
                stages=self.stages,
                batches=self.batch_request,
                total_replications=1,
                context=self.context,
                max_instances=self.max_instances
            )
            self.configs[config.id] = config
        return None
