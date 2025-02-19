import subprocess
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from logger import setup_logger
from utils import lazy_import, is_tail_running

log = logging.getLogger("app")


@dataclass(frozen=True)
class TestClassContainer:
    module: str
    test_class: str
    
    @cached_property
    def impl(self):
        return lazy_import(self.module, self.test_class)
    

class TestClass(TestClassContainer, Enum):
    get = ("get", "Get")
    test = ("testing.test", "Test")
    subtest = ("testing.subtest", "Subtest")
    cross_model = ("testing.cross_model", "CrossModel")
    intra_model = ("testing.intra_model", "IntraModel")
    sample_splitting = ("testing.sample_splitting", "SampleSplitting")



if __name__ == "__main__":
    setup_logger()

    # Optional -- Opens a new terminal/shell w/ live logs
    repo_path = Path(__file__).parents[1]
    if not is_tail_running():
        if sys.platform == "darwin":
            subprocess.Popen([
                "osascript",
                "-e",
                f'tell application "Terminal" to do script "cd {repo_path} && tail -f logs/app.log"'
            ])
        elif sys.platform == "win32":
            subprocess.Popen([
                "start",
                "cmd",
                "/k",
                f'cd /d {repo_path} && tail -f logs/app.log'
            ], shell=True)