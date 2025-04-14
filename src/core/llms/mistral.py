"""
Mistral LLM module.

Contains the Mistral model.
"""
from core.llms.clients.mistral import MistralClient
from _types.llm_config import LLMConfig



class Mistral(MistralClient):
    """
    Mistral Model (inherits the MistralClient).
    """
    def _translate_config(self, config: LLMConfig):
        return {
            "max_completion_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "random_seed": config.seed,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty
        }
