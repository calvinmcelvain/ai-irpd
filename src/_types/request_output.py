"""
Module for RequestOut and MetaOut models.
"""
import json
from dataclasses import dataclass
from pydantic import BaseModel

from _types.meta_output import MetaOut
from _types.prompts import Prompts



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