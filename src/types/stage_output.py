"""
Stage output model.
"""
from pathlib import Path
from typing import List
from dataclasses import dataclass, field

from types.irpd_request import IRPDRequest


@dataclass
class StageOutput:
    stage_name: str
    subset: str
    llm_str: str
    replication: int
    batch_id: str = None
    batch_path: Path = None
    sub_path: Path = None
    outputs: List[IRPDRequest] = field(default_factory=list)
    complete: bool = False
