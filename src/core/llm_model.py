"""
LLM aggregated module.

Aggregates all LLM models. Structure strongly follows:
https://github.com/TIGER-AI-Lab/MEGA-Bench/blob/main/megabench/models/model_type.py
"""
from enum import Enum
from typing import Literal
from dataclasses import dataclass, field
from functools import cached_property

from helpers.utils import lazy_import, get_env_var



@dataclass(frozen=True)
class LLMModelClassContainer:
    module: str
    model_class: str

    @cached_property
    def impl(self):
        model_class = lazy_import(self.module, self.model_class)
        return model_class


class LLMModelClass(LLMModelClassContainer, Enum):
    GPT = ("core.llms.gpt", "GPT")
    CLAUDE = ("core.llms.claude", "Claude")
    GEMINI = ("core.llms.gemini", "Gemini")
    NOVA = ("core.llms.nova", "Nova")
    MISTRAL = ("core.llms.mistral", "Mistral")
    GROK = ("core.llms.grok", "Grok")


@dataclass(frozen=True)
class OtherArgs:
    json_tool: bool = None
    region: str = None
    base_url: str = None
    batches: bool = True


@dataclass(frozen=True)
class LLMModelContainer:
    key: str
    model_name: str
    api_key: str
    model_class: LLMModelClass
    other_args: OtherArgs = field(default=OtherArgs())


class LLMModel(LLMModelContainer, Enum):
    GPT_4O_0806 = (
        "GPT_4O_0806",
        "gpt-4o-2024-08-06",
        "OPENAI_API_KEY",
        LLMModelClass.GPT
    )
    GPT_4O_1120 = (
        "GPT_4O_1120",
        "gpt-4o-2024-11-20",
        "OPENAI_API_KEY",
        LLMModelClass.GPT
    )
    GPT_4O_MINI_0718 = (
        "GPT_4O_MINI_0718",
        "gpt-4o-mini-2024-07-18",
        "OPENAI_API_KEY",
        LLMModelClass.GPT,
        OtherArgs(json_tool=True)
    )
    GPT_O1_1217 = (
        "GPT_O1_1217",
        "o1-2024-12-17",
        "OPENAI_API_KEY",
        LLMModelClass.GPT
    )
    GPT_O1_MINI_0912 = (
        "GPT_O1_MINI_0912",
        "o1-mini-2024-09-12",
        "OPENAI_API_KEY",
        LLMModelClass.GPT,
        OtherArgs(json_tool=True)
    )
    GPT_O3_MINI_0131 = (
        "GPT_O3_MINI_0131",
        "o3-mini-2025-01-31",
        "OPENAI_API_KEY",
        LLMModelClass.GPT
    )
    GROK_2_1212 = (
        "GROK_2_1212",
        "grok-2-1212",
        "XAI_API_KEY",
        LLMModelClass.GROK,
        OtherArgs(base_url="https://api.x.ai/v1")
    )
    CLAUDE_3_5_SONNET = (
        "CLAUDE_3_5_SONNET",
        "claude-3.5-sonnet-20241022",
        "ANTHROPIC_API_KEY",
        LLMModelClass.CLAUDE,
        OtherArgs(json_tool=True)
    )
    CLAUDE_3_7_SONNET = (
        "CLAUDE_3_7_SONNET",
        "claude-3-7-sonnet-20250219",
        "ANTHROPIC_API_KEY",
        LLMModelClass.CLAUDE,
        OtherArgs(json_tool=True)
    )
    GEMINI_2_FLASH = (
        "GEMINI_2_FLASH",
        "gemini-2.0-flash-001",
        "GOOGLE_API_KEY",
        LLMModelClass.GEMINI,
        OtherArgs(batches=False)
    )
    GEMINI_2_FLASH_LITE = (
        "GEMINI_2_FLASH_LITE",
        "gemini-2.0-flash-lite-001",
        "GOOGLE_API_KEY",
        LLMModelClass.GEMINI,
        OtherArgs(batches=False)
    )
    GEMINI_1_5_PRO = (
        "GEMINI_1_5_PRO",
        "gemini-1.5-pro-002",
        "GOOGLE_API_KEY",
        LLMModelClass.GEMINI,
        OtherArgs(batches=False)
    )
    NOVA_PRO_V1 = (
        "NOVA_PRO_V1",
        "amazon.nova-pro-v1:0",
        "BEDROCK_API_KEY",
        LLMModelClass.NOVA,
        OtherArgs(json_tool=True, region="us-east-1", batches=False)
    )
    MISTRAL_LARGE_2411 = (
        "MISTRAL_LARGE_2411",
        "mistral-large-2411",
        "MISTRAL_API_KEY",
        LLMModelClass.MISTRAL
    )
    
    def get_llm_instance(
        self,
        config: Literal["base", "res1", "res2", "res3"] = "base",
        print_response: bool = False
    ):
        """
        Creates an instance of a LLM model.

        Args:
            config (str, optional): The config used for LLM model. Defaults to 
            "base". Can be from ["base", "res1", "res2", "res3"].
            print_response (bool, optional): If True, prints response of all
            LLM requests. Defaults to False.
        """
        model_class = self.model_class.impl
        return model_class(
            api_key=get_env_var(self.api_key),
            model=self.model_name,
            configs=config,
            print_response=print_response,
            json_tool=self.other_args.json_tool,
            region=self.other_args.region,
            base_url=self.other_args.base_url,
            batches=self.other_args.batches
        )