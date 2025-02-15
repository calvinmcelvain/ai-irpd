import logging
from testing.irpd_base import IRPDBase
from stages import *

log = logging.getLogger("app.cross_validation")


class CrossValidation(IRPDBase):
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
        self._test_type = {"cross_validation"}
    
    def _generate_test_paths(self):
        pass
    
    def _generate_test_configs(self):
        pass
    
    def run(self, max_instances = None, threshold = 0.5, config_ids = None):
        super().run(max_instances, threshold, config_ids)