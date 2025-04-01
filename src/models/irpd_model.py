import logging
from pathlib import Path
from typing import List, Union, Literal
from dataclasses import dataclass
from enum import Enum
from functools import cached_property

from utils import lazy_import, load_config


CONFIGS = load_config("irpd_configs.yml")
DEFAULTS = CONFIGS["defaults"]

CASES = Literal["uni", "switch", "uniresp", "first", "uni_switch"]
RAS = Literal["ra1", "ra2", "both"]
TREATMENTS = Literal["imperfect", "perfect", "merged"]
STAGES = Literal["1", "1r", "1c", "2", "3"]
LLMS = Literal[
    "GPT_4O_0806", "GPT_4O_1120", "GPT_4O_MINI_0718", "GPT_O1_1217",
    "GPT_O1_MINI_0912", "GPT_O3_MINI_0131", "GROK_2_1212", 
    "CLAUDE_3_5_SONNET", "CLAUDE_3_7_SONNET", "GEMINI_2_FLASH",
    "GEMINI_1_5_PRO", "NOVA_PRO_V1", "MISTRAL_LARGE_2411"
]
LLM_CONFIGS = Literal["base", "res1", "res2", "res3"]


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
    
    def get_irpd_instance(
        self,
        cases: Union[List[CASES], CASES],
        ras: Union[List[RAS], RAS],
        treatments: Union[List[TREATMENTS], TREATMENTS],
        stages: Union[List[STAGES], STAGES],
        llms: Union[List[LLMS], LLMS] = DEFAULTS["llms"],
        llm_configs: Union[List[LLM_CONFIGS], LLM_CONFIGS] = DEFAULTS["llm_configs"],
        batch: bool = False,
        test_paths: Union[List[Union[str, Path]], Union[str, Path]] = None,
        **kwargs
    ):
        test_class = self.impl
        return test_class(
            cases=cases,
            ras=ras,
            treatments=treatments,
            stages=stages,
            llms=llms,
            llm_configs=llm_configs,
            batch=batch,
            test_paths=test_paths,
            **kwargs
        )