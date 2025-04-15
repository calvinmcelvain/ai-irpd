from typing import List
from dataclasses import dataclass

from _types.request_output import RequestOut



@dataclass
class BatchResponse:
    response_id: str
    response: RequestOut


@dataclass
class BatchOut:
    batch_id: str
    responses: List[BatchResponse]