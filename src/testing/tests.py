import logging
from itertools import product
from testing.irpd_base import IRPDBase
from test_config import TestConfig

log = logging.getLogger("app.testing.tests")


class Tests(IRPDBase):
    def __init__(
        self,
        case,
        ras,
        treatments,
        stages,
        test: str = "test",
        llms = ["GPT_4O_1120"],
        llm_configs = ["base"],
        project_path = None,
        new_test = True
    ):
        super().__init__(
            case,
            ras,
            treatments,
            stages,
            llms,
            llm_configs,
            project_path,
            new_test
        )
        self.test = test
        self._prod_lcrt = list(product(
            self.llms, self.llm_configs, self.ras, self.treatments
        ))
        
        if not new_test and len(self._prod_lcrt) > 1:
            log.warning(
                "Cannot have multiple test configs w/ previous test",
                " Defaulted 'new_test' to True."
            )
            self.new_test = True
        self._generate_test_configs()
    
    def _generate_test_path(self):
        if self.test == "test":
            test_dir = self.output_path / self.case
            prefix = "test_"
        else:
            test_dir = self.output_path / "_subtests"
            prefix = ""
        
        current_test = self._get_max_test_number(test_dir, prefix)
        
        if not self.new_test:
            return [test_dir / f"{prefix}{current_test}"]
        
        return [
            test_dir / f"{prefix}{test + current_test}"
            for test, _ in enumerate(self._prod_lcrt, start=1)
        ]
    
    def _generate_test_configs(self):
        for idx, llm, llm_config, ra, treatment in enumerate(self.prod_lcrt):
            config = TestConfig(
                case=self.case,
                ra=ra,
                treatment=treatment,
                llm=self._generate_model_instance(llm, llm_config),
                llm_config=llm_config,
                stages=self.stages,
                test_type=self.test,
                test_path=self.test_paths[idx]
            )
            self.TEST_CONFIGS[config.test_id] = config
    
    def run(self, max_instances = None, threshold = 0.5):
        for config in self.TEST_CONFIGS.values():
            log.info(f"TEST: Start {self.test.upper()} | {config.test_id}")
            
            path = config.test_path
            if not path.exists():
                short_path = path.relative_to(self.project_path)
                log.info(f"TEST: Making test directory: {short_path}...")
                config.test_path.mkdir(exist_ok=True)
                log.info(f"TEST: Created test directory: {path.exists()}")
            
            for stage_name in self.stages:
                context = self.outputs.get(config.test_id, 1)
                stage_instance = globals().get(f"Stage{stage_name}")(
                    config, path, context, max_instances, threshold
                )
                    
                output = stage_instance.run()
                self.OUTPUTS.store(config.test_id, 1, output)
        log.info(f"Test: End of {self.test.upper()} | {config.test_id}")