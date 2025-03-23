from pydantic import BaseModel
from typing import Optional
from models.meta_output import MetaOutput


class RequestOut(BaseModel):
    text: str
    meta: Optional[MetaOutput]
    parsed: Optional[BaseModel]