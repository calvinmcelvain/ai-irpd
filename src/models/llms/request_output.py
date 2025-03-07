from pydantic import BaseModel
from models.llms.meta_output import MetaOutput


class RequestOut(BaseModel):
    text: str
    meta: MetaOutput
    parsed: BaseModel = None