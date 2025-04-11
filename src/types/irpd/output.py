"""
Test outputs module.

Contains the TestOutput & StageOutput objects. Also the TestMeta object (along
with its composition objects).
"""
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from types.request_output import RequestOut



@dataclass
class StageOutput:
    stage_name: str
    subset: str
    llm_str: str
    replication: int
    batch_id: str = None
    batch_path: Path = None
    outputs: List[RequestOut] = field(default_factory=list)
    complete: bool = False


@dataclass
class TestOutput:
    llm_str: str
    stage_outputs: List[StageOutput] = field(default_factory=list)
    complete: bool = False
    
    def check_test_complete(self):
        """
        Sets the complete field based on whether all the complete fields for 
        all StageOutput objects are True.
        """
        self.complete = all(output.complete for output in self.stage_outputs)
        

# TestMeta objects.
class ModelInfo(BaseModel):
    model: str
    parameters: dict


class SubsetInfo(BaseModel):
    created: str = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class StageInfo(BaseModel):
    subsets: Dict[str, SubsetInfo] = {}
    batch_id: Optional[str] = None
    batch_path: Optional[str] = None


class TestMeta(BaseModel):
    model_info: ModelInfo
    test_info: dict
    stages: Dict[str, StageInfo] = {}
