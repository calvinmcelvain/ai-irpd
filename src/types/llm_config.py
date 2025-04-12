from dataclasses import dataclass



@dataclass
class LLMConfig:
    """
    Generic LLM config.
    """
    max_tokens: int
    temperature: float
    top_p: float
    seed: int
    frequency_penalty: float
    presence_penalty: float
    reasoning: str
