from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from _types.request_output import RequestOut



@dataclass
class IRPDOutput:
    request_out: RequestOut
    subset: str
    response_path: Optional[Path]
    user_path: Optional[Path]
    system_path: Optional[Path]
    
