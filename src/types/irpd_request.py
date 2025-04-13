"""
Contains IRPDRequest model.
"""
from dataclasses import dataclass
from pathlib import Path

from types.request_output import RequestOut



@dataclass
class IRPDRequest:
    output: RequestOut
    prompt_id: str = None
    prompt_path: Path = None
    response_path: Path = None
