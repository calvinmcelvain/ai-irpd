"""
Prompts Model.
"""
from pydantic import BaseModel


class Prompts(BaseModel):
    system: str
    user: object
