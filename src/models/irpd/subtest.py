import logging
from typing import Union, Optional, List
from pathlib import Path

from models.irpd.test import Test


log = logging.getLogger(__name__)



class Subtest(Test):
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
        test_type: str = "subtest"
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
            test_paths,
            test_type
        )
    
    def _generate_test_paths(self):
        if self.test_paths:
            return self._validate_test_paths()
        test_paths = []
        test_dir = self.output_path / "subtests"
        current_test = self._get_max_test_number(test_dir, "")
        paths = [test_dir / f"{i + 1 + current_test}" for i in range(len(self._prod))]
        test_paths.extend(paths)
        return test_paths