from typing import Dict
from pathlib import Path
from dataclasses import dataclass

from helpers.utils import load_json_n_validate
from _types.irpd_meta import IRPDMeta
from _types.stage_output import StageOutput



@dataclass
class TestOutput:
    sub_path: Path
    meta_path: Path
    replication: int
    llm_str: str
    stage_outputs: Dict[str, StageOutput]
    meta: IRPDMeta
    complete: bool = False
    
    def __post_init__(self):
        # Updating meta if exists after intialization
        if self.meta_path.exists():
            self.meta = load_json_n_validate(self.meta_path, IRPDMeta)
    
