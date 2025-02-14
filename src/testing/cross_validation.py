import logging
from testing.irpd_base import IRPDBase

log = logging.getLogger("app.testing.cross_validation")


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
    
    def _generate_test_path(self):
        pass
    
    def _generate_test_configs(self):
        pass
    
    def run(self, max_instances = None, threshold = 0.5):
        pass