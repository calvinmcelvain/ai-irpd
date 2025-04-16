from dataclasses import dataclass
from typing import List, Optional, Tuple
from uuid import uuid4



@dataclass
class IRPDConfig:
    case: str
    ra: str
    treatment: str
    llms: List[str]
    llm_config: str
    test_type: str
    test_path: str
    data_path: str
    prompts_path: str
    stages: List[str]
    batches: bool
    total_replications: int
    context: Optional[Tuple[int, int]]
    max_instances: Optional[int]
    cases: List[str] = None
    id: str = uuid4().hex
    
    def __post_init__(self):
        # A case can be a composition of cases, defined with a `_` (e.g., 
        # 'uni_switch'). This is accounted for via a `cases` field.
        self.cases = self.case.split("_")