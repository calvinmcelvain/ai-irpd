"""
XAI's Grok module.

Defines general Grok configs.
"""
from core.llms.clients.openai import OpenAIClient
from types.llm_config import LLMConfig


class Grok(OpenAIClient):
    """
    Grok model (inherits OpenAIClient model).
    """
    def _translate_config(self, config: LLMConfig):
        return {
            "max_completion_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "seed": config.seed,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty
        }