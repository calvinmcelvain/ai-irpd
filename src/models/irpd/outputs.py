import logging
from typing import Dict, List
from dataclasses import dataclass, field

from models.irpd.test_configs import TestConfig
from models.request_output import RequestOut


log = logging.getLogger(__name__)



@dataclass
class StageOutput:
    stage: str
    subset: str
    outputs: Dict[RequestOut] = field(default_factory=dict)


@dataclass
class SubOutput:
    llm_str: str
    replication: int
    stage_outputs: StageOutput


@dataclass
class TestOutput:
    config: TestConfig
    stage_outputs: Dict[str, List[StageOutput]] = field(default_factory=dict)
    