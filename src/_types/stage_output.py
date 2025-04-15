from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from _types.irpd_output import IRPDOutput


@dataclass
class StageOutput:
    stage_name: str
    stage_path: Path
    outputs: Dict[str, List[IRPDOutput]]
    complete: bool = False
