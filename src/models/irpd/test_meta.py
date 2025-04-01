from typing import Dict, List, Optional
from pydantic import BaseModel

from models.irpd.test_config import TestConfig



class ModelInfo(BaseModel):
    model: str
    parameters: BaseModel


class StageTokens(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class StageInfo(BaseModel):
    created: str
    subsets: List[str]
    tokens: Dict[str, StageTokens]
    batch_id: Optional[str] = None


class TestMeta(BaseModel):
    model_info: ModelInfo
    test_info: TestConfig
    stages: Dict[str, StageInfo]