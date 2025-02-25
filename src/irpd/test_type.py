import logging
from typing import List
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from utils import lazy_import

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestClassContainer:
    module: str
    test_class: str
    
    @cached_property
    def impl(self):
        return lazy_import(self.module, self.test_class)
    

class TestClass(TestClassContainer, Enum):
    TEST = ("irpd.test", "Test")
    SUBTEST = ("irpd.subtest", "Subtest")
    CROSS_MODEL = ("irpd.cross_model", "CrossModel")
    INTRA_MODEL = ("irpd.intra_model", "IntraModel")
    SAMPLE_SPLITTING = ("irpd.sample_splitting", "SampleSplitting")
    
    def get_type_instance(
        self,
        case: str,
        ras: List[str] | str,
        treatments: List[str] | str,
        stages: List[str],
        **kwargs
    ):
        test_class = self.impl
        return test_class(
            case=case,
            ras=ras,
            treatments=treatments,
            stages=stages,
            **kwargs
        )