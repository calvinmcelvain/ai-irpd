from dataclasses import dataclass
from pathlib import Path

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