from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

@dataclass
class TestConfig:
    case: str
    ra: str
    treatment: str
    llms: list[str]
    llm_config: str
    test_type: str
    test_path: Path
    stages: list[str]
    max_instances: int | None
    id: str = None
    
    def __post_init__(self):
        self.id = uuid4().hex