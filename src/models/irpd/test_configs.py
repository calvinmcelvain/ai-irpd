from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from uuid import uuid4

from models.llms.base_llm import BaseLLM



@dataclass
class TestConfig:
    case: str
    ra: str
    treatment: str
    llms: List[str]
    llm_config: str
    test_type: str
    test_path: Path
    data_path: Path
    prompts_path: Path
    stages: List[str]
    batches: bool
    total_replications: int
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
    

@dataclass
class SubConfig(TestConfig):
    sub_path: Path
    llm_str: str
    llm_instance: BaseLLM
    replication: int
    meta_path: Path = None
    
    def __post_init__(self):
        self.meta_path = self.sub_path / "_test_meta.json"
        if self.batches:
            self.batches = self.llm_instance.batches

    
@dataclass
class StageConfig(SubConfig):
    stage_name: str
    subset: str = None
    stage_path: Path = None
    prompts_path: Path = None
    responses_path: Path = None
    
    def __post_init__(self):
        self.stage_path = self.sub_path / f"stage_{self.stage_name}"
        self.prompts_path = self.stage_path / self.subset / "prompts"
        self.responses_path = self.stage_path / self.subset / "responses"
    
    