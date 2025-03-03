import logging
from irpd.test import Test

log = logging.getLogger(__name__)


class Subtest(Test):
    def __init__(
        self,
        case,
        ras,
        treatments,
        stages,
        llms = None,
        llm_configs = None,
        project_path = None,
        new_test = True,
        test_paths = None,
        test = "subtest"
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
            test_paths,
            test
        )