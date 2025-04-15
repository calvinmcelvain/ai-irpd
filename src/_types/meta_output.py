from dataclasses import dataclass
from typing import Dict
from datetime import datetime



@dataclass
class MetaOut:
    input_tokens: int
    output_tokens: int
    model: str
    configs: Dict
    total_tokens: int = None
    created: int = None
    
    def __post_init__(self):
        self.total_tokens = sum([self.input_tokens, self.output_tokens])
        
        # For standardization, the created field is written as a unix timestamp.
        if not self.created:
            self.created = int(datetime.now().timestamp())