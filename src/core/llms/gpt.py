"""
OpenAI's GPT module.

Defines general configs for GPT model.
"""
from core.llms.clients.openai import OpenAIClient
from types.llm_config import LLMConfig



class GPT(OpenAIClient):
    """
    GPT model (inherits OpenAIClient model).
    """
    _reasoning_models = {
        "o1-2024-12-17",
        "o1-mini-2024-09-12",
        "o3-mini-2025-01-31"
    }
    
    def _translate_config(self, config: LLMConfig):
        configs = {
            "max_completion_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "seed": config.seed,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty
        }
        if self.model in self._reasoning_models:
            configs.update({"reasoning_effort": config.reasoning})
        return configs
        
    
    def _prep_messages(self, user: str, system: str):
        # For o1 (and above) models, a `developer` role is used instead of 
        # standard `system` role.
        if self.model in self._reasoning_models:
            developer = {"role": "developer", "content": system}
            return {"messages": [developer, self._prep_user_message(user)]}
        else:
            return super()._prep_messages(user, system)