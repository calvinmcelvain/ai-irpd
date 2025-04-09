"""
OpenAI's GPT module.

Defines general configs for GPT model.
"""
from pydantic import BaseModel, Field

from models.llms.openai_client import OpenAIClient



class GPTConfigs(BaseModel):
    max_completion_tokens: int = Field(None, ge=1, le=4096)
    reasoning_effort: str = Field(None, pattern="^(low|medium|high)$")
    temperature: float = Field(None, ge=0, le=1)
    top_k: int = Field(None, ge=0)
    top_p: float = Field(None, ge=0, le=1)
    seed: int = Field(None, ge=1)
    frequency_penalty: float = Field(None, ge=0, le=1)
    presence_penalty: float = Field(None, ge=0, le=1)


class GPT(OpenAIClient):
    """
    GPT model (inherits OpenAIClient model).
    """
    configs: GPTConfigs
    
    def default_configs(self):
        return GPTConfigs()
    
    def _prep_messages(self, user: str, system: str):
        # For o1 (and above) models, a `developer` role is used instead of 
        # standard `system` role.
        if "o1" in self.model:
            developer = {"role": "developer", "content": system}
            return {"messages": [developer, self._prep_user_message(user)]}
        else:
            return super()._prep_messages(user, system)