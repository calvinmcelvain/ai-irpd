"""
Module for Prompts object.
"""
from pydantic import BaseModel


class Prompts(BaseModel):
    system: str
    user: object