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
    
    def __init__(
        self,
        api_key=None,
        model=None,
        confgis=None,
        print_response=False,
        json_tool=False,
        base_url=None
    ):
        super().__init__(api_key, model, confgis, print_response, json_tool, base_url)
        self.base_url = "https://api.x.ai/v1"
    
    def default_configs(self):
        return GrokConfigs()