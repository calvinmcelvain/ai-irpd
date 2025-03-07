from dataclasses import dataclass
from datetime import datetime
from models.prompts import Prompts


@dataclass
class MetaOutput:
    input_tokens: int
    output_tokens: int
    prompt: Prompts
    created: int = None
    total_tokens: int = None
    
    def __post_init__(self):
        self.total_tokens = sum([self.input_tokens, self.output_tokens])
        if not self.created:
            self.created = int(datetime.now().timestamp())