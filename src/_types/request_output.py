"""
Module for RequestOut and MetaOut models.
"""
import json
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel

from _types.prompts import Prompts



@dataclass
class MetaOut:
    input_tokens: int
    output_tokens: int
    model: str
    configs: dict
    total_tokens: int = None
    created: int = None
    
    def __post_init__(self):
        self.total_tokens = sum([self.input_tokens, self.output_tokens])
        
        # For standardization, the created field is written as a unix timestamp.
        if not self.created:
            self.created = int(datetime.now().timestamp())


@dataclass
class RequestOut:
    parsed: BaseModel
    text: str = None
    prompts: Prompts = None
    meta: MetaOut = None
    
    def __post_init__(self):
        # If RequestOut object initialized w/ structured output, text field
        # is filled.
        if self.parsed and not self.text:
            self.text = json.dumps(self.parsed.model_dump(), indent=4)