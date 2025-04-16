from pydantic import BaseModel
from typing import Dict, Optional

from _types.irpd_config import IRPDConfig



class SubsetInfo(BaseModel):
    created: str = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class StageInfo(BaseModel):
    subsets: Dict[str, SubsetInfo] = {}
    batch_id: Optional[str] = None
    batch_path: Optional[str] = None


class IRPDMeta(BaseModel):
    model: Optional[str]
    configs: Optional[Dict]
    test_info: IRPDConfig
    stages: Dict[str, StageInfo] = {}
