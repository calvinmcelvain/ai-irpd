from typing import Dict, List
from pathlib import Path
from dataclasses import dataclass
from models.request_output import RequestOut


@dataclass
class StageOutput:
    stage: str
    outputs: Dict[str, List[RequestOut]]
    
    def write(self, path: Path):
        pass