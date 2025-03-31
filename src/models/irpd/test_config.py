from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path
from uuid import uuid4



@dataclass
class TestConfig:
    case: str
    ra: str
    treatment: str
    llms: List[str]
    llm_config: str
    test_type: str
    test_path: Path
    test_number: Union[str, int]
    stages: List[str]
    max_instances: Optional[int] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        self.id = uuid4().hex