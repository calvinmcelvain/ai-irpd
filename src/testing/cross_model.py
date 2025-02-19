import logging
from itertools import product
from testing.irpd_base import IRPDBase
from test_config import TestConfig
from testing.stages import *

log = logging.getLogger("app.cross_model")


class CrossModel(IRPDBase):
    def __init__(
        self,
        case,
        ras,
        treatments,
        stages,
        N: int,
        llms = ["GPT_4O_1120"],
        llm_configs = ["base"],
        project_path = None,
        new_test = True,
        test_paths = None
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
        self._test_type = "cross_model"
        if N > 1:
            self.N = N
        else:
            log.warning("N must be greater than or equal to 1. Defaulted to 1")
            self.N = 1
        self._prod_crt = list(product(
            self.llm_configs, self.ras, self.treatments
        ))
        
        if not new_test and len(self._prod_crt) > 1:
            log.warning(
                "Cannot have multiple test configs w/ previous test",
                " Defaulted 'new_test' to True."
            )
            self.new_test = True
        
        if self.test_paths and not len(self.test_paths) == len(self._prod_crt):
            log.error(
                "test_paths must be the same length as the number of test configs."
            )
            raise ValueError
        self._generate_configs()
    
    def _generate_test_paths(self):
        test_dir = self.output_path / self._test_type
        current_test = self._get_max_test_number(test_dir)
        
        if not self.new_test:
            return [test_dir / f"test_{current_test}"]
        
        return [
            test_dir / f"test_{test + current_test}"
            for test, _ in enumerate(self._prod_crt, start=1)
        ]
    
    def _generate_configs(self):
        if not self.test_paths:
            self.test_paths = self._generate_test_paths()
        for idx, prod in enumerate(self._prod_crt):
            llm_config, ra, treatment = prod
            config = TestConfig(
                case=self.case,
                ra=ra,
                treatment=treatment,
                llms=self.llms,
                llm_config=llm_config,
                stages=self.stages,
                test_type=self._test_type,
                test_path=self.test_paths[idx]
            )
            self.configs[config.test_id] = config
    
    def run(self, max_instances = None, threshold = 0.5, config_ids = None):
        super().run(max_instances, threshold, config_ids)
        for config in self._test_configs.values():
            log.info(f"TEST: Start of CROSS-MODEL Test = {config.test_id}")
            
            path = config.test_path
            if not path.exists():
                short_path = path.relative_to(self.project_path)
                log.info(f"TEST: Making test directory: {short_path}...")
                config.test_path.mkdir(exist_ok=True)
                log.info(f"TEST: Created test directory: {path.exists()}")
            
            for l in self.llms:
                log.info(f"TEST: Start {l} replications")
                
                llm = self._generate_model_instance(l, config.llm_config)
                
                for n in range(1, self.N + 1):
                    sub_path = path / l / f"replication_{n}"
                    
                    log.info(f"START: {l} replicate {n}")
                    for stage_name in self.stages:
                        context = self.OUTPUTS.get(config.test_id, n, l)
                        stage_instance = globals().get(f"Stage{stage_name}")(
                            config, sub_path, context, llm, max_instances, threshold
                        )
                        
                        stage_instance.run()
                        self.OUTPUTS.store(config.test_id, n, l, stage_instance.output)
                    log.info(f"TEST: End {l} replicate {n}")
                log.info(f"TEST: End of {l} replications")
            log.info(f"Test: End of CROSS-MODEL Test = {config.test_id}")