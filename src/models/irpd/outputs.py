import logging
from typing import Dict, List
from dataclasses import dataclass, field

from models.irpd.test_configs import TestConfig, SubConfig, StageConfig
from models.request_output import RequestOut


log = logging.getLogger(__name__)



@dataclass
class StageOutput:
    stage_config: StageConfig
    stage: str
    subset: str
    stage_outputs: List[RequestOut] = field(default_factory=list)


@dataclass
class SubOutput:
    sub_config: SubConfig
    llm_str: str
    replication: int
    sub_outputs: List[StageOutput] = field(default_factory=list)


@dataclass
class TestOutput:
    config: TestConfig
    test_outputs: List[SubOutput] = field(default_factory=list)
    