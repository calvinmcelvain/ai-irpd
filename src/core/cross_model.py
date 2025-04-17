"""
Cross-model IRPD test module.

Contains the CrossModel model.
"""
import logging
from itertools import product
from typing import Optional, List, Union, Tuple
from pathlib import Path

from core.base import IRPDBase
from _types.irpd_config import IRPDConfig


log = logging.getLogger("app")



class CrossModel(IRPDBase):
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
            max_summaries,
            output_path,
            prompts_path,
            data_path,
            test_paths,
            batch,
            test_type
        )
        # The number of tests for cross-model tests is the total combinations
        # of LLM configs, cases, ras, & treatments.
        self._prod = list(product(
            self.llm_configs, self.cases, self.ras, self.treatments
        ))
        
        self.test_paths = self._generate_test_paths()
        self._generate_configs()
    
    def _generate_test_paths(self):
        if self.test_paths:
            return self._validate_test_paths()

        # Test paths are in directory: .../outputs/cross_model/
        test_dir = self.output_path / self.test_type
        current_test = self._get_max_test_number(test_dir)
        test_paths = [test_dir / f"test_{i + 1 + current_test}" for i in range(len(self._prod))]
        return test_paths
    
    def _generate_configs(self):
        for idx, prod in enumerate(self._prod):
            llm_config, case, ra, treatment = prod
            config = IRPDConfig(
                case=case,
                ra=ra,
                treatment=treatment,
                llms=self.llms,
                llm_config=llm_config,
                test_type=self.test_type,
                test_path=self.test_paths[idx].as_posix(),
                data_path=self.data_path.as_posix(),
                prompts_path=self.prompts_path.as_posix(),
                stages=self.stages,
                batches=self.batch_request,
                total_replications=self.replications,
                context=self.context,
                max_instances=self.max_instances,
            )
            self.configs[config.id] = config
        return None
    