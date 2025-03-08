from typing import Dict
from dataclasses import dataclass
from models.irpd.stage_output import StageOutput


@dataclass
class TestOutput:
    id: str
    llm: str
    replication: int
    stages: Dict[str, StageOutput]