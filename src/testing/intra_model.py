import logging
from itertools import product
from src.testing.irpd import IRPD
from test_config import TestConfig

log = logging.getLogger("app.testing.intra_model")


class IntraModel(IRPD):
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
        test_dir = self.output_path / "replication"
        current_test = self._get_max_test_number(test_dir)
        
        if not self.new_test:
            return [test_dir / f"test_{current_test}"]
        
        return [
            test_dir / f"test_{test + current_test}"
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
                test_type="replication",
                test_path=self.test_paths[idx]
            )
            self.TEST_CONFIGS[config.test_id] = config
    
    def run(self, max_instances = None, threshold = 0.5):
        for config in self.TEST_CONFIGS.values():
            log.info(f"TEST: Start Replication Test | {config.test_id}")
            
            path = config.test_path
            if not path.exists():
                short_path = path.relative_to(self.project_path)
                log.info(f"TEST: Making test directory: {short_path}...")
                config.test_path.mkdir(exist_ok=True)
                log.info(f"TEST: Created test directory: {path.exists()}")
            
            for n in range(1, self.N + 1):
                sub_path = path / f"replication_{n}"
                
                log.info(f"START: Replicate {n}")
                for stage_name in self.stages:
                    context = self.outputs.get(config.test_id, n)
                    stage_instance = globals().get(f"Stage{stage_name}")(
                        config, sub_path, context, max_instances, threshold
                    )
                    
                    output = stage_instance.run()
                    self.OUTPUTS.store(config.test_id, n, output)
                log.info(f"TEST: End Replicate {n}")
            log.info(f"Test: End of Replication Test | {config.test_id}")