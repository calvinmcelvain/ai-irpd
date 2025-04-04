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
    outputs: List[RequestOut] = field(default_factory=dict)


@dataclass
class SubOutput:
    sub_config: SubConfig
    llm_str: str
    replication: int
    stage_outputs: List[StageOutput]


@dataclass
class TestOutput:
    config: TestConfig
    outputs: List[SubOutput] = field(default_factory=dict)
    