"""
IRPD config module.

Contains the IRPDConfig model & it's dataclass methods.
"""
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path
from uuid import uuid4



@dataclass
class IRPDConfig:
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
    max_instances: Optional[int] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        # A config ID is a UUID
        self.id = uuid4().hex
        
        # A case can be a composition of cases, defined with a `_` (e.g., 
        # 'uni_switch'). This is accounted for via a `cases` field.
        self.cases = self.case.split("_")
    
    def convert_to_dict(self):
        """
        Converts all fields that are Path objects into a string.
        
        Used for writing meta data, as problems arrise when tryint to write 
        Path objects in JSON.
        """
        return asdict(self, dict_factory=lambda x: {
            k: (v.as_posix() if isinstance(v, Path) else v) for k, v in x
        })