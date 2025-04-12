"""
Google's Gemini module.

Contains Gemini model.
"""
from core.llms.clients.genai import GenAIClient
from types.llm_config import LLMConfig



class Gemini(GenAIClient):
    """
    Google's Gemini model (inherits the GenAICLient class).
    """
    def _translate_config(self, config: LLMConfig):
        return {
            "max_output_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "seed": config.seed,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty
        }