import logging
from itertools import product
from pathlib import Path
from logger import clear_logger
from utils import str_to_list
from irpd.irpd_base import IRPDBase
from irpd.test_config import TestConfig
from irpd.stages import *

log = logging.getLogger(__name__)


class Test(IRPDBase):
    def __init__(
        self,
        case,
        ras,
        treatments,
        stages,
        llms = ["GPT_4O_1120"],
        llm_configs = ["base"],
        project_path = None,
        new_test = True,
        test_paths = None,
        test = "test"
    ):
        super().__init__(
            case,
            ras,
            treatments,
            stages,
            llms,
            llm_configs,
            project_path,
            new_test,
            test_paths
        )
        self._test_type = test
        self._prod_lcrt = list(product(
            self.llms, self.llm_configs, self.ras, self.treatments
        ))
        
        if not new_test and len(self._prod_lcrt) > 1:
            log.warning(
                "Cannot have multiple test configs w/ previous test",
                " Defaulted 'new_test' to True."
            )
            self.new_test = True
        if self.test_paths:
            test_paths = str_to_list(self.test_paths)
            if not all(isinstance(path, Path) for path in test_paths):
                test_paths = [Path(path) for path in test_paths]
            if not len(self.test_paths) == len(self._prod_lcrt):
                log.error(
                    "test_paths must be the same length as the number of test configs."
                )
                raise ValueError
        self._generate_configs()
    
    def _generate_test_paths(self):
        if self._test_type == "test":
            test_dir = self.output_path / "base_tests" / self.case
            prefix = "test_"
        else:
            test_dir = self.output_path / "subtests"
            prefix = ""
        
        current_test = self._get_max_test_number(test_dir, prefix)
        
        if not self.new_test:
            return [test_dir / f"{prefix}{current_test}"]
        
        return [
            test_dir / f"{prefix}{test + current_test}"
            for test, _ in enumerate(self._prod_lcrt, start=1)
        ]
    
    def _generate_configs(self):
        if not self.test_paths:
            self.test_paths = self._generate_test_paths()
        for idx, prod in enumerate(self._prod_lcrt):
            llm, llm_config, ra, treatment = prod
            config = TestConfig(
                case=self.case,
                ra=ra,
                treatment=treatment,
                llms=llm,
                llm_config=llm_config,
                stages=self.stages,
                test_type=self._test_type,
                test_path=self.test_paths[idx]
            )
            self.configs[config.test_id] = config
    
    def run(self, max_instances = None, threshold = 0.5, config_ids = None):
        super().run(max_instances, threshold, config_ids)
        for config in self._test_configs.values():
            clear_logger(app=False)
            log.info(f"TEST: Start {self._test_type.upper()} = {config.test_id}")
            
            llm = self._generate_model_instance(config.llms, config.llm_config)
            
            path = config.test_path
            if not path.exists():
                short_path = path.relative_to(self.project_path)
                log.info(f"TEST: Making test directory: {short_path}...")
                config.test_path.mkdir(exist_ok=True)
                log.info(f"TEST: Created test directory: {path.exists()}")
            
            for stage_name in self.stages:
                context = self.OUTPUTS.get(config.test_id, 1, config.llms)
                stage_instance = globals().get(f"Stage{stage_name}")(
                    config, path, context, llm, max_instances, threshold
                )
                    
                stage_instance.run()
                self.OUTPUTS.store(config.test_id, 1, config.llms, stage_instance.output)
        log.info(f"TEST: End of {self._test_type.upper()} = {config.test_id}")