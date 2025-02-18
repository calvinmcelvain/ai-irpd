import logging
from testing.tests import Test

log = logging.getLogger("app.subtests")


class Subtest(Test):
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