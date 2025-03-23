from typing import Dict, List
from pathlib import Path
from dataclasses import dataclass, field
from models.request_output import RequestOut


@dataclass
class StageOutput:
    stage: str
    outputs: Dict[str, List[RequestOut]] = field(default_factory=dict)
    
    def write(self, path: Path):
        pass