from pydantic import BaseModel
from typing import Optional



class LLMConfig(BaseModel):
    """
    Generic LLM config.
    """
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    seed: Optional[int]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    reasoning: Optional[str]
