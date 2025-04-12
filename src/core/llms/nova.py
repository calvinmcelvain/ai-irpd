"""
Nova module.

Defines Nova configs.
"""
from core.llms.clients.bedrock import BedrockClient
from types.llm_config import LLMConfig

    

class Nova(BedrockClient):
    """`
    Nova class (inherits BedrockClient).
    """
    def _translate_config(self, config: LLMConfig):
        return {
            "max_new_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p
        }