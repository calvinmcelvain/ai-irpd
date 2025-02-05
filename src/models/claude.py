from pydantic import BaseModel, Field
from models.anthropic_client import AnthropicClient


class ClaudeConfigs(BaseModel):
    max_tokens: int = Field(None, ge=1, le=4096)
    temperature: float = Field(None, ge=0, le=1)
    top_k: int = Field(None, ge=0)
    top_p: float = Field(None, ge=0, le=1)
    

class Claude(AnthropicClient):
    configs: ClaudeConfigs
    
    def default_configs(self):
        return ClaudeConfigs()