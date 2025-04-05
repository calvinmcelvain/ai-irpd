from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from uuid import uuid4



@dataclass
class TestConfig:
    case: str = None
    ra: str = None
    treatment: str = None
    llms: List[str] = None
    llm_config: str = None
    test_type: str = None
    test_path: Path = None
    data_path: Path = None
    prompts_path: Path = None
    stages: List[str] = None
    batches: bool = None
    total_replications: int = None
    cases: List[str] = None
    instance_types: List[str] = None
    max_instances: Optional[int] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        self.id = uuid4().hex
        self.cases = self.case.split("_")
        
        if self.case in {"uni", "uniresp"}:
            self.instance_types = ["ucoop", "udef"]
        else:
            self.instance_types =["coop", "def"]
    
    