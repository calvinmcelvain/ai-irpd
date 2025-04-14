from typing import Dict
from pathlib import Path
from dataclasses import dataclass

from _types.irpd_meta import IRPDMeta
from _types.stage_output import StageOutput



@dataclass
class TestOutput:
    sub_path: Path
    meta_path: Path
    replication: int
    llm_str: str
    meta: IRPDMeta
    stage_outputs: Dict[str, StageOutput]
    
