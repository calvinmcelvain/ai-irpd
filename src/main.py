import logging
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from logger import setup_logger
from utils import lazy_import

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
    test = ("testing.tests", "Tests")
    subtest = ("testing.tests", "Tests")
    inter_model = ("testing.inter_model", "InterModel")
    intra_model = ("testing.intra_model", "IntraModel")
    cross_validation = ("testing.cross_validation", "CrossValidation")


if __name__ == "__main__":
    setup_logger()
