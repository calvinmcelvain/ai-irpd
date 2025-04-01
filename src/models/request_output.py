import json
from dataclasses import dataclass
from pydantic import BaseModel
from models.meta_output import MetaOutput


@dataclass
class RequestOut:
    text: str = None
    parsed: BaseModel = None
    meta: MetaOutput = None
    
    def __post_init__(self):
        if self.parsed and not self.text:
            self.text = json.dumps(self.parsed.model_dump())