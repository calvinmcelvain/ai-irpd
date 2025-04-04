from typing import Dict, List, Optional
from pydantic import BaseModel

from models.irpd.test_configs import SubConfig



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