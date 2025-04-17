from typing import List
from dataclasses import dataclass

from _types.irpd_output import IRPDOutput



@dataclass
class BatchResponse:
    response_id: str
    response: IRPDOutput


@dataclass
class BatchOut:
    batch_id: str
    responses: List[BatchResponse]