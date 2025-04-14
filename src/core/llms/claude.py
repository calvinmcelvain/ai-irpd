"""
Module for Claude model.

Defines general configs for Claude models.
"""
from core.llms.clients.anthropic import AnthropicClient
from _types.llm_config import LLMConfig

    

class Claude(AnthropicClient):
    """
    Claude model (inherits Anthropic model).
    """
    def _translate_config(self, config: LLMConfig):
        return {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p
        }