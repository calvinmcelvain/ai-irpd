from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

@dataclass
class TestConfig:
    case: str
    ra: str
    treatment: str
    llm: str
    llm_instance: object
    test_type: str
    test_path: Path
    project_path: Path
    stages: list[str]
    print_response: bool
    max_instances: int | None
    id: str = None
    
    def __post_init__(self):
        self.id = uuid4().hex