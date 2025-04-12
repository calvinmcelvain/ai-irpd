"""
Contains IRPDPrompts model.
"""
from dataclasses import dataclass
from pathlib import Path

from types.prompts import Prompts



@dataclass
class IRPDPrompts:
    prompts: Prompts
    prompt_id: str
    prompt_path: Path
    response_path: Path
