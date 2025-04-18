"""
Nova module.

Defines Nova configs.
"""
from pydantic import BaseModel, Field

from models.llms.bedrock_client import BedrockClient



class NovaConfigs(BaseModel):
    max_new_tokens: int = Field(None, ge=1, le=5000)
    temperature: float = Field(None, ge=0, le=1)
    top_p: float = Field(None, ge=0, le=1)
    

class Nova(BedrockClient):
    """`
    Nova class (inherits BedrockClient).
    """
    configs: NovaConfigs
    
    def default_configs(self):
        return NovaConfigs()