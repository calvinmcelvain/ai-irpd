from pydantic import BaseModel
from models.meta_output import MetaOutput


class RequestOut(BaseModel):
    text: str
    meta: MetaOutput
    parsed: BaseModel = None