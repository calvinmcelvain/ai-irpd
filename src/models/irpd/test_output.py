from typing import Dict
from dataclasses import dataclass, field
from models.irpd.stage_output import StageOutput


@dataclass
class TestOutput:
    id: str
    llm: str
    replication: int
    stage_outputs: Dict[str, StageOutput] = field(default_factory=dict)