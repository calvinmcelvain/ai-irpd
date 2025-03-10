import logging
from typing import List, Union
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
    

class IRPDTestClass(TestClassContainer, Enum):
    TEST = ("models.irpd.test", "Test")
    SUBTEST = ("models.irpd.subtest", "Subtest")
    CROSS_MODEL = ("models.irpd.cross_model", "CrossModel")
    INTRA_MODEL = ("models.irpd.intra_model", "IntraModel")
    SAMPLE_SPLITTING = ("models.irpd.sample_splitting", "SampleSplitting")
    
    def get_type_instance(
        self,
        case: Union[List[str], str],
        ras: Union[List[str], str],
        treatments: Union[List[str], str],
        stages: Union[List[str], str],
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