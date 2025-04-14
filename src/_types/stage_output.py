"""
Stage output model.
"""
from pathlib import Path
from typing import List
from dataclasses import dataclass, field

from _types.request_output import RequestOut


@dataclass
class StageOutput:
    stage_name: str
    subset: str
    llm_str: str
    replication: int
    output_path: Path
    batch_id: str = None
    batch_path: Path = None
    outputs: List[RequestOut] = field(default_factory=list)
    complete: bool = False
