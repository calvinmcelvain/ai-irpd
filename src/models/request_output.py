import json
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel

from models.prompts import Prompts



@dataclass
class MetaOutput:
    input_tokens: int
    output_tokens: int
    total_tokens: int = None
    created: int = None
    
    def __post_init__(self):
        self.total_tokens = sum([self.input_tokens, self.output_tokens])
        if not self.created:
            self.created = int(datetime.now().timestamp())


@dataclass
class RequestOut:
    text: str = None
    parsed: BaseModel = None
    prompts: Prompts
    meta: MetaOutput = None
    
    def __post_init__(self):
        if self.parsed and not self.text:
            self.text = json.dumps(self.parsed.model_dump())