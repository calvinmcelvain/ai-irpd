from dataclasses import dataclass
from datetime import datetime
from models.prompts import Prompts


@dataclass
class MetaOutput:
    input_tokens: int
    output_tokens: int
    prompt: Prompts
    created: int = None
    
    def __post_init__(self):
        if not self.created:
            self.created = int(datetime.now().timestamp())