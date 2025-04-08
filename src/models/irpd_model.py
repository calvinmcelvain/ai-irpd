"""
IRPD aggregated module.

Aggregates the IRPD test models: Test, Subtest, IntraModel, CrossModel,
and SampleSplitting.
"""
import logging
from pathlib import Path
from typing import List, Union, Literal, Optional
from dataclasses import dataclass
from enum import Enum
from functools import cached_property

from utils import lazy_import, load_config, to_list


CONFIGS = load_config("irpd_configs.yml")
DEFAULTS = CONFIGS["defaults"]

# Specifying the arguments for test models.
# Also makes it easier when creating an instance w/ autofill.
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
    TEST = ("models.irpd.irpd_test", "Test")
    SUBTEST = ("models.irpd.irpd_subtest", "Subtest")
    CROSS_MODEL = ("models.irpd.irpd_cross_model", "CrossModel")
    INTRA_MODEL = ("models.irpd.irpd_intra_model", "IntraModel")
    SAMPLE_SPLITTING = ("models.irpd.irpd_sample_splitting", "SampleSplitting")
    
    def get_irpd_instance(
        self,
        cases: Union[List[CASES], CASES],
        ras: Union[List[RAS], RAS],
        treatments: Union[List[TREATMENTS], TREATMENTS],
        stages: Union[List[STAGES], STAGES],
        N: int = 1,
        llms: Union[List[LLMS], LLMS] = DEFAULTS["llms"],
        llm_configs: Union[List[LLM_CONFIGS], LLM_CONFIGS] = DEFAULTS["llm_configs"],
        max_instances: Optional[int] = None,
        batch: bool = False,
        test_paths: Union[List[Union[str, Path]], Union[str, Path]] = None,
        **kwargs
    ):
        """
        Creates an IRPD test instance.

        Args:
            cases (Union[List[CASES], CASES]): The cases that want to be run.
            Can be from ["uni", "uniresp", "switch", "first", 
            "uni_switch"].
            ras (Union[List[RAS], RAS]): The RA summaries to be used (if want
            LLM generated summaries, i.e., stage 0, use `exp`). Can be from
            ["ra1", "ra2", "both", "exp"].
            treatments (Union[List[TREATMENTS], TREATMENTS]): The treatments to
            be run. Can be from ["imperfect", "perfect", "merged"].
            stages (Union[List[STAGES], STAGES]): The stages to be run for each
            test. Can be from ["0", "1", "1r", "1c", "2", "3"]. Must be on order
            if Test or Subtest.
            N (int, optional): The number of replications. Only used if test
            type is IntraModel or CrossModel. Defaults to 1.
            llms (Union[List[LLMS], LLMS], optional): The LLM models to be used
            in tests. Defaults to `GPT_4O_1120`.
            llm_configs (Union[List[LLM_CONFIGS], LLM_CONFIGS], optional): The
            configs used for LLM model(s). Defaults to `base`. Can be from 
            ["base", "res1", "res2", "res3"].
            max_instances (Optional[int], optional): The maximum number of 
            "instances" or "summaries" to be used in iterative stages ("0", "2",
            and "3"). Defaults to None (all instances used).
            batch (bool, optional): If True, then Batch API used for tests, if
            the LLM supports it. Defaults to False.
            test_paths (Union[List[Union[str, Path]], Union[str, Path]], 
            optional): The specific paths to used for tests. Generally this is
            used if continuing stopped test or adding more stages to a test. 
            Defaults to None.
            kwargs:
                - prompts_path, output_path: The paths to be used for outputs
                and/or prompts.
        """
        # Adjust stages only for Test and Subtest models.
        if self in {IRPDTestClass.TEST, IRPDTestClass.SUBTEST}:
            stage_list = STAGES.__args__
            if len(to_list(stages)) == 1 and stage_list.index(to_list(stages)[0]) != 0:
                stages = list(stage_list[:stage_list.index(to_list(stages)[0]) + 1])
        
        test_class = self.impl
        return test_class(
            cases=cases,
            ras=ras,
            treatments=treatments,
            stages=stages,
            N=N,
            llms=llms,
            llm_configs=llm_configs,
            max_instances=max_instances,
            batch=batch,
            test_paths=test_paths,
            **kwargs
        )