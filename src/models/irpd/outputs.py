import logging
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from models.request_output import RequestOut
from models.irpd.test_configs import SubConfig, StageConfig


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
    batch_path: Path = None


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
    llm_str: str = None
    replication: int = None
    
    def __post_init__(self):
        self.llm_str = self.stage_config.llm_str
        self.replication = self.stage_config.replication


@dataclass
class TestOutput:
    llm_str: str
    stage_outputs: List[StageOutput] = field(default_factory=list)
    complete: bool = False
    
    def test_complete(self):
        self.complete = all(output.complete for output in self.stage_outputs)