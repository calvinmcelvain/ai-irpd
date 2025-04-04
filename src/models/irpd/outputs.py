import logging
from typing import List
from dataclasses import dataclass, field

from models.irpd.test_configs import TestConfig, SubConfig, StageConfig
from models.request_output import RequestOut


log = logging.getLogger(__name__)



@dataclass
class StageOutput:
    stage_config: StageConfig
    stage_name: str
    subset: str
    outputs: List[RequestOut] = field(default_factory=list)
    complete: bool = False


@dataclass
class SubOutput:
    sub_config: SubConfig
    llm_str: str
    replication: int
    stage_outputs: List[StageOutput] = field(default_factory=list)
    complete: bool = False


@dataclass
class TestOutput:
    config: TestConfig
    test_outputs: List[SubOutput] = field(default_factory=list)
    complete: bool = False
    