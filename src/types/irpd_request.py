"""
Contains IRPDRequest model.
"""
from dataclasses import dataclass
from pathlib import Path

from types.prompts import Prompts
from types.request_output import RequestOut



@dataclass
class IRPDRequest:
    prompts: Prompts
    prompt_id: str
    prompt_path: Path
    response_path: Path
    output: RequestOut
