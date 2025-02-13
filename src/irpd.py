import re
import logging
from typing import List, Dict
from pathlib import Path
from itertools import product
from importlib import reload
import utils, models, test_config, output_manager, logger, stages
from utils import get_env_var
from models import LLMModel
from test_config import TestConfig
from output_manager import OutputManager
from logger import setup_logger
from stages import *

if __name__ == "__main__":
    reload(utils)
    reload(models)
    reload(test_config)
    reload(output_manager)
    reload(logger)
    reload(stages)
    setup_logger()

log = logging.getLogger("app.irpd")
    

class IRPD:
    _VALID_STAGES = ['0', '1', '1r', '1c', '2', '3']
    _VALID_CASES = ['uni', 'uniresp', 'switch', 'first', 'uni_switch']
    _VALID_RAS = ['thi', 'eli', 'both', 'exp']
    _VALID_TEST_TYPES = ["test", "subtest", "replication", "cross_model_validation"]
    _VALID_TREATMENTS = ['noise', 'no_noise', 'merged']
    _VALID_LLMS = LLMModel._member_names_
    _VALID_LLM_CONFIGS = ["base", "res1", "res2", "res3"]
    
    OUTPUTS: OutputManager = OutputManager()
    TEST_CONFIGS: Dict[str, TestConfig]

    def __init__(
        self, 
        case: str,
        ras: List[str],
        treatments: List[str],
        stages: List[str],
        test_type: str = "test",
        llms: List[str] = ["GPT_4O_1120"],
        llm_configs: List[str] = ["base"],
        project_path: str = None,
        new_test: bool = True
    ):
        self.case = self._validate_arg(
            [case], self._VALID_CASES, "cases")[0]
        self.ras = self._validate_arg(
            ras, self._VALID_RAS, "ras")
        self.treatments = self._validate_arg(
            treatments, self._VALID_TREATMENTS, "treatments")
        self.stages = self._validate_arg(
            stages, self._VALID_STAGES, "stages")
        self.test_type = self._validate_arg(
            [test_type], self._VALID_TEST_TYPES, "test_type")[0]
        self.llms = self._validate_arg(
            llms, self._VALID_LLMS, "llms")
        self.llm_configs = self._validate_arg(
            llm_configs, self._VALID_LLM_CONFIGS, "llm_configs")
        self.project_path = Path(
            project_path if project_path else get_env_var("PROJECT_DIRECTORY")
        )
        self.output_path = self.project_path / "output"
        self.new_test = new_test
        self.product_rtcl = list(product(self.ras, self.treatments, self.llm_configs, self.llms))
        self._generate_test_configs()

    def _validate_arg(self, arg: list[str], valid_values: list[str], name: str):
        if not isinstance(arg, list) or not all(isinstance(item, str) for item in arg):
            raise ValueError(f"{name} must be a list of strings.")

        valid_set = set(valid_values)
        index_map = {value: i for i, value in enumerate(valid_values)} 

        valid_items, invalid_items = [], []
        for item in arg:
            (valid_items if item in valid_set else invalid_items).append(item)

        if not valid_items:
            log.error(
                f"All provided `{name}` values are invalid. No valid items remain. "
                f"Allowed values: {valid_values}"
            )
            raise ValueError
        
        if invalid_items:
            log.warning(
                f"Some `{name}` values are invalid and were ignored: {invalid_items}. "
                f"Allowed values: {valid_values}"
            )

        return sorted(valid_items, key=lambda x: index_map[x])
    
    def _generate_model_instance(self, llm: str, config: str):
        return getattr(LLMModel, llm).get_model_instance(
                config=config, print_response=self.print_response
        )
    
    @staticmethod
    def _get_max_test_number(directory: Path, prefix: str):
        pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
        return max(
            map(int, (match.group(1) for p in directory.iterdir() if (match := pattern.match(p.name)))),
            default=0
        )

    def _generate_test_path(self):
        t = self.test_type
        test_dirs = {
            "test": (self.output_path / self.case, "test_"),
            "subtest": (self.output_path / "_subtests", ""),
            "replication": (self.output_path / "replication_tests", "test_"),
            "cross_model_validation": (self.output_path / "cross_model_validation", "test_"),
        }

        base_dir, prefix = test_dirs[t]
        base_dir.mkdir(parents=True, exist_ok=True)
        
        test_num = self._get_max_test_number(base_dir, prefix)
        
        if t in {"cross_model_validation"}:
            n = 0
            if self.new_test:
                new_test = (len(self.test_configs) % len(self.llms)) == 0
                n = len(set(i.test_path for i in self.test_configs)) + (1 if new_test else 0)
            return base_dir / f"test_{test_num + n}"
        
        if t in {"test", "subtest", "replication"}:
            k = any(stage in {"0", "1"} for stage in self.stages)
            if not self.new_test and len(self.product_rtcl) > 1:
                self.new_test = True
                log.warning(
                    f"`new_test` must be True if multiple tests are specified."
                    " `new_test` defaulted to True."
                )
            if self.new_test and not k and t not in {"replication"}:
                log.error(f"Stages must contain '0' or '1' for new {self.test_type}.")
                raise ValueError("Invalid stage for test type")
            n = len(set(i.test_path for i in self.test_configs)) + (1 if self.new_test else 0)
            return base_dir / f"{prefix}{test_num + n}"
    
    @staticmethod
    def _generate_sub_path(test_config: TestConfig, n: int):
        if test_config.test_type in {"test", "subtest"}:
            return test_config.test_path
        if test_config.test_path in {"replication"}:
            return test_config.test_path / f"replication_{n}"
        if test_config.test_type in {"cross_model_validation"}:
            return test_config.test_path / test_config.llm.model / f"replicaiton_{n}"
        
    def _generate_test_configs(self):
        for ra, treatment, llm_config, llm in self.product_rtcl:
            config = TestConfig(
                case=self.case,
                ra=ra,
                treatment=treatment,
                llm=self._generate_model_instance(llm, llm_config),
                stages=self.stages,
                test_type=self.test_type,
                test_path=self._generate_test_path(),
            )
            self.TEST_CONFIGS[config.test_id] = config
    
    def run(
        self,
        max_instances: int = None,
        N: int = 1,
        threshold: float = 0.5
    ):
        for test_config in self.test_configs.values():
            log.info(f"START: Test {test_config.test_id}")
            
            path = test_config.test_path
            if not path.exists():
                short_path = path.relative_to(self.project_path)
                log.info(f"DIRECTORY: Making test directory: {short_path}...")
                test_config.test_path.mkdir(exist_ok=True)
                log.info(f"DIRECTORY: Created: {path.exists()}")
            
            for n in range(1, N + 1):
                sub_path = self._generate_sub_path(test_config, n)
                
                log.info(f"START: Replicate {n}")
                for stage_name in self.stages:
                    log.info(f"START: Stage {stage_name}")
                    
                    context = self.outputs.get(test_config.test_id, n)
                    stage_instance = globals().get(f"Stage{stage_name}")(
                        test_config, sub_path, context, max_instances, threshold
                    )
                    
                    output = stage_instance.run()
                    self.OUTPUTS.store(test_config.test_id, n, output)
                    
                    log.info(f"END: Stage {stage_name}")
                log.info(f"END: Replicate {n}")
            log.info(f"END: Test {test_config.test_id}")