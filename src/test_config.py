from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

@dataclass
class TestConfig:
    case: str
    ra: str
    treatment: str
    llm: object
    test_type: str
    test_path: Path
    stages: list[str]
    max_instances: int | None
    test_id: str = None
    
    def __post_init__(self):
        self.test_id = uuid4().hex