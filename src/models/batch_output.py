"""
Batch output module.

Contains the dataclass objects for batch request outputs (BatchOut & BatchResponse).
"""
from typing import List
from dataclasses import dataclass

from models.request_output import RequestOut



@dataclass
class BatchResponse:
    response_id: str
    response: RequestOut


@dataclass
class BatchOut:
    batch_id: str
    responses: List[BatchResponse]