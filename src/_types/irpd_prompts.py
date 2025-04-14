from dataclasses import dataclass

from _types.prompts import Prompts



@dataclass
class IRPDPrompts:
    prompts: Prompts
    prompt_id: str
    
