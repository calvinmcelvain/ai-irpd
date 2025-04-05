import logging
from pydantic import BaseModel
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from models.request_output import RequestOut
from models.irpd.test_configs import TestConfig, SubConfig, StageConfig


log = logging.getLogger(__name__)



class ModelInfo(BaseModel):
    model: str
    parameters: BaseModel


class StageTokens(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class StageInfo(BaseModel):
    created: str = None
    subsets: List[str] = []
    tokens: Dict[str, StageTokens]
    batch_id: Optional[str] = None


class TestMeta(BaseModel):
    model_info: ModelInfo
    sub_config: SubConfig
    stages: Dict[str, StageInfo]


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
    batch_id: str = None


@dataclass
class TestOutput:
    config: TestConfig
    test_outputs: List[SubOutput] = field(default_factory=list)
    complete: bool = False
    