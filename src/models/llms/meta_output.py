from pydantic import BaseModel
from models.prompts import Prompts


class MetaOutput(BaseModel):
    input_tokens: int
    output_tokens: int
    created: str
    prompt: Prompts
    