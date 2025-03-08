from typing import Dict
from pathlib import Path
from dataclasses import dataclass
from models.llms.request_output import RequestOut


@dataclass
class StageOutput:
    stage: str
    outputs: Dict[str, RequestOut]
    
    def write(self, path: Path):
        pass