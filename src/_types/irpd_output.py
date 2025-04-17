import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from dataclasses import dataclass

from helpers.utils import write_file, write_json
from _types.prompts import Prompts



@dataclass
class IRPDOutput:
    parsed: BaseModel
    subset: str
    prompts: Optional[Prompts]
    response_path: Optional[Path]
    user_path: Optional[Path]
    system_path: Optional[Path]
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    text: str = None
    
    def __post_init__(self):
        self.total_tokens = sum([self.input_tokens, self.output_tokens])
        
        # If object initialized w/ structured output, text field is filled.
        if self.parsed and not self.text:
            self.text = json.dumps(self.parsed.model_dump(), indent=4)
        
        if not self.created:
            self.created = int(datetime.now().timestamp())
    
    def write(self) -> None:
        """
        Writes the output & responses to paths (if defined).
        """
        if all(
            self.text,
            self.prompts,
            self.response_path,
            self.user_path,
            self.system_path
        ):
            prompt_paths = [self.user_path, self.system_path]
            prompt_writes = [self.prompts.user, self.prompts.system]
            write_file(prompt_paths, prompt_writes)
            
            # writes in json form if paresed.
            if self.parsed:
                write_json(self.response_path, self.parsed.model_dump())
            else:
                write_file(self.response_path, self.text)
            
            return None
        
        