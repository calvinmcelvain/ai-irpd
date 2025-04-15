from pydantic import BaseModel
from typing import Dict, Optional

from _types.irpd_config import IRPDConfig



class SubsetInfo(BaseModel):
    created: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


class StageInfo(BaseModel):
    subsets: Dict[str, SubsetInfo]
    batch_id: Optional[str]
    batch_path: Optional[str]


class IRPDMeta(BaseModel):
    model: Optional[str]
    configs: Optional[Dict]
    test_info: IRPDConfig = None
    stages: Dict[str, StageInfo] = None
