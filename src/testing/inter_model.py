import logging
from itertools import product
from testing.irpd_base import IRPDBase
from test_config import TestConfig

log = logging.getLogger("app.testing.inter_model")


class InterModel(IRPDBase):
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
    
    def _generate_test_path(self):
        test_dir = self.output_path / "cross_model_validation"
        current_test = self._get_max_test_number(test_dir)
        
        if not self.new_test:
            return [test_dir / f"test_{current_test}"]
        
        return [
            test_dir / f"test_{test + current_test}"
            for test, _ in enumerate(self._prod_crt, start=1)
        ]
    
    def _generate_test_configs(self):
        for idx, llm_config, ra, treatment in enumerate(self.prod_crt):
            config = TestConfig(
                case=self.case,
                ra=ra,
                treatment=treatment,
                llm=self.llms,
                llm_config=llm_config,
                stages=self.stages,
                test_type="replication",
                test_path=self.test_paths[idx]
            )
            self.TEST_CONFIGS[config.test_id] = config
    
    def run(self, max_instances = None, threshold = 0.5):
        for config in self.TEST_CONFIGS.values():
            log.info(f"TEST: Start Cross-Model Validation Test | {config.test_id}")
            
            path = config.test_path
            if not path.exists():
                short_path = path.relative_to(self.project_path)
                log.info(f"TEST: Making test directory: {short_path}...")
                config.test_path.mkdir(exist_ok=True)
                log.info(f"TEST: Created test directory: {path.exists()}")
            
            for l in self.llms:
                log.info(f"TEST: Start {l} replications")
                
                llm = self._generate_model_instance(l, config.llm_config)
                llm_config = config
                llm_config.llm = llm
                
                for n in range(1, self.N + 1):
                    sub_path = path / l / f"replication_{n}"
                    
                    log.info(f"START: {l} replicate {n}")
                    for stage_name in self.stages:
                        context = self.outputs.get(config.test_id, n)
                        stage_instance = globals().get(f"Stage{stage_name}")(
                            llm_config, sub_path, context, max_instances, threshold
                        )
                        
                        output = stage_instance.run()
                        self.OUTPUTS.store(config.test_id, n, output)
                    log.info(f"TEST: End {l} replicate {n}")
                log.info(f"TEST: End of {l} replications")
            log.info(f"Test: End of Replication Test | {config.test_id}")