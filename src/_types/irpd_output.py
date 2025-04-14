from pathlib import Path
from dataclasses import dataclass

from _types.request_output import RequestOut



@dataclass
class IRPDOutput:
    request_out: RequestOut
    response_path: Path
    user_path: Path
    system_path: Path
    
