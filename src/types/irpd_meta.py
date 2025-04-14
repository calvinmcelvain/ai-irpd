"""
IRPD meta  module.

Contains IRPDMeta model (along with its composition objects).
"""
from pydantic import BaseModel
from typing import Dict, Optional

from types.irpd_config import IRPDConfig
from types.llm_config import LLMConfig



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
    model: str
    configs: LLMConfig
    test_info: IRPDConfig = None
    stages: Dict[str, StageInfo] = {}
