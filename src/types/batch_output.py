"""
Models for Batch outputs.
"""
from typing import List
from pydantic import BaseModel

from types.request_output import RequestOut



class BatchResponse(BaseModel):
    response_id: str
    response: RequestOut


class BatchOut(BaseModel):
    batch_id: str
    responses: List[BatchResponse]