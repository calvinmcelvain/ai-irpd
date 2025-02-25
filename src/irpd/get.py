import logging
from pathlib import Path
from typing import List
from output_manager import OutputManager, StageRun
from llms.base_model import RequestOut
from utils import get_env_var, load_json

log = logging.getLogger("app.get")


class Get:
    _VALID_TEST_TYPES = ["subtest", "test", "intra-model", "cross-model"]
    
    def __init__(
        self,
        test_type: str,
        tests: int | List[int],
        case: str = None,
        llm: str = None,
        project_path: str = None
    ):
        self.test_type = test_type
        self.tests = tests
        self.case = case
        self.project_path = Path(
            project_path if project_path else get_env_var("PROJECT_DIRECTORY")
        )
        self.llm = llm
        self.output_path = self.project_path / "output"
        self._validate_args()
        self.test_dirs = self._fetch_test_dirs()
        self.outputs = OutputManager()
    
    def _validate_args(self):
        if not self.test_type in self._VALID_TEST_TYPES:
            msg = f"Test type not accepted: {self.test_type}"
            log.error(msg)
            raise ValueError(msg)
        if self.test_type == "test" and not self.case:
            msg = "'case' must be specified for test type: 'test'"
            log.error(msg)
            raise ValueError(msg)
        if self.test_type == "intra-model" and not self.llm:
            log.warning(
                "'llm' must be specified for test type: 'intra-model'."
                " Defaulted to GPT_4O_0806"
            )
            self.llm = "GPT_4O_0806"
        if not isinstance(self.tests, list):
            self.tests = [self.tests]
        
    def _fetch_test_dirs(self):
        test_parent_map = {
            "subtest": "subtests",
            "test": f"base_tests/{self.case}",
            "intra-model": "intra_model",
            "cross-model": "cross_model"
        }
        parent_path = self.output_path / test_parent_map[self.test_type]
        prefix = "test_" if self.test_type != "subtest" else ""
        test_dirs = []
        for test in self.tests:
            test_path = parent_path / f"{prefix}{test}"
            if test_path.exists():
                test_dirs.append(test_path)
            else:
                log.warning(f"Test path does not exist: {test_path}")
        return test_dirs
    
    def get_outputs(self):
        def process_stage(stage, stage_run):
            stage_name = stage.name.split("stage_")[1]
            if stage_name == "1c":
                for i in stage.iterdir():
                    if i.is_dir():
                        path = i / f"stg_{stage_name}_{i.name}_response.txt"
                        if path.exists():
                            response = RequestOut(response=load_json(path, True))
                            stage_run.store("combined", i.name, response)
            else:
                for c in stage.iterdir():
                    if c.is_dir():
                        for i in c.iterdir():
                            if i.is_dir():
                                path = i / f"stg_{stage_name}_{i.name}_response.txt"
                                if path.exists():
                                    response = RequestOut(response=load_json(path, True))
                                    stage_run.store(c.name, i.name, response)

        for test_dir in self.test_dirs:
            if self.test_type == "intra-model":
                for rep in test_dir.iterdir():
                    if rep.is_dir():
                        rep_num = int(rep.name.split("replication_")[1])
                        for stage in rep.iterdir():
                            if stage.name.startswith("stage"):
                                stage_run = StageRun(stage.name.split("stage_")[1])
                                process_stage(stage, stage_run)
                                self.outputs.store(test_dir.name, rep_num, self.llm, stage_run)
            elif self.test_type == "cross-model":
                for llm in test_dir.iterdir():
                    if llm.is_dir():
                        for rep in llm.iterdir():
                            if rep.is_dir():
                                rep_num = int(rep.name.split("replication_")[1])
                                for stage in rep.iterdir():
                                    if stage.name.startswith("stage"):
                                        stage_run = StageRun(stage.name.split("stage_")[1])
                                        process_stage(stage, stage_run)
                                        self.outputs.store(test_dir.name, rep_num, llm.name, stage_run)
            elif self.test_type in {"test", "subtest"}:
                for stage in test_dir.iterdir():
                    if stage.name.startswith("stage"):
                        stage_run = StageRun(stage.name.split("stage_")[1])
                        process_stage(stage, stage_run)
                        self.outputs.store(test_dir.name, 1, self.llm, stage_run)
