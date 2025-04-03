from dataclasses import dataclass
from typing import List, Optional
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
    stages: List[str]
    max_instances: Optional[int] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        self.id = uuid4().hex


@dataclass
class SubConfig(TestConfig):
    sub_path: Path
    llm: str
    batches: bool
    total_replications: int
    
    
@dataclass
class StageConfig(SubConfig):
    stage_name: str
    prompts_path: Path = None
    responses_path: Path = None
    
    def __post_init__(self):
        stage_path = self.sub_path / f"stage_{self.stage_name}"
        self.prompts_path = stage_path / "prompts"
        self.responses_path = stage_path / "responses"
    
    