from pydantic import BaseModel, Field

from models.llms.openai_client import OpenAIClient



class GrokConfigs(BaseModel):
    max_completion_tokens: int = Field(None, ge=1, le=4096)
    temperature: float = Field(None, ge=0, le=1)
    top_p: float = Field(None, ge=0, le=1)
    seed: int = Field(None, ge=1)
    frequency_penalty: float = Field(None, ge=0, le=1)
    presence_penalty: float = Field(None, ge=0, le=1)


class Grok(OpenAIClient):
    configs: GrokConfigs
    
    def default_configs(self):
        return GrokConfigs()