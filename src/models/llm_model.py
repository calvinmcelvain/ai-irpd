# Structure strongly follows
# https://github.com/TIGER-AI-Lab/MEGA-Bench/blob/main/megabench/models/model_type.py
from enum import Enum
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from utils import lazy_import, load_config, validate_json, get_env_var

CONFIGS = load_config("llm_configs.yml")


@dataclass(frozen=True)
class ModelClassContainer:
    module: str
    model_class: str
    model_config: str

    @cached_property
    def impl(self):
        model_class = lazy_import(self.module, self.model_class)
        model_configs = lazy_import(self.module, self.model_config)
        return model_class, model_configs


class ModelClass(ModelClassContainer, Enum):
    GPT = ("llms.gpt", "GPT", "GPTConfigs")
    CLAUDE = ("llms.claude", "Claude", "ClaudeConfigs")
    GEMINI = ("llms.gemini", "Gemini", "GeminiConfigs")
    NOVA = ("llms.nova", "Nova", "NovaConfigs")
    MISTRAL = ("llms.mistral", "Mistral", "MistralConfigs")
    GROK = ("llms.grok", "Grok", "GrokConfigs")


@dataclass(frozen=True)
class OtherArgs:
    json_tool: bool = None
    region: str = None


@dataclass(frozen=True)
class LLMModelContainer:
    key: str
    model_name: str
    api_key: str
    model_class: ModelClass
    other_args: OtherArgs = field(default=OtherArgs())


class LLMModel(LLMModelContainer, Enum):
    GPT_4O_0806 = (
        "GPT_4O_0806",
        "gpt-4o-2024-08-06",
        "OPENAI_API_KEY",
        ModelClass.GPT
    )
    GPT_4O_1120 = (
        "GPT_4O_1120",
        "gpt-4o-2024-11-20",
        "OPENAI_API_KEY",
        ModelClass.GPT
    )
    GPT_4O_MINI_0718 = (
        "GPT_4O_MINI_0718",
        "gpt-4o-mini-2024-07-18",
        "OPENAI_API_KEY",
        ModelClass.GPT,
        OtherArgs(json_tool=True)
    )
    GPT_O1_1217 = (
        "GPT_O1_1217",
        "o1-2024-12-17",
        "OPENAI_API_KEY",
        ModelClass.GPT
    )
    GPT_O1_MINI_0912 = (
        "GPT_O1_MINI_0912",
        "o1-mini-2024-09-12",
        "OPENAI_API_KEY",
        ModelClass.GPT,
        OtherArgs(json_tool=True)
    )
    GPT_O3_MINI_0131 = (
        "GPT_O3_MINI_0131",
        "o3-mini-2025-01-31",
        "OPENAI_API_KEY",
        ModelClass.GPT
    )
    GROK_2_1212 = (
        "GROK_2_1212",
        "grok-2-1212",
        "XAI_API_KEY",
        ModelClass.GROK
    )
    CLAUDE_3_5_SONNET = (
        "CLAUDE_3_5_SONNET",
        "claude-3.5-sonnet-20241022",
        "ANTHROPIC_API_KEY",
        ModelClass.CLAUDE,
        OtherArgs(json_tool=True)
    )
    CLAUDE_3_7_SONNET = (
        "CLAUDE_3_7_SONNET",
        "claude-3-7-sonnet-20250219",
        "ANTHROPIC_API_KEY",
        ModelClass.CLAUDE,
        OtherArgs(json_tool=True)
    )
    GEMINI_2_FLASH = (
        "GEMINI_2_FLASH",
        "gemini-2.0-flash-001",
        "GOOGLE_API_KEY",
        ModelClass.GEMINI
    )
    GEMINI_2_FLASH_LITE = (
        "GEMINI_2_FLASH_LITE",
        "gemini-2.0-flash-lite-001",
        "GOOGLE_API_KEY",
        ModelClass.GEMINI
    )
    GEMINI_1_5_PRO = (
        "GEMINI_1_5_PRO",
        "gemini-1.5-pro-002",
        "GOOGLE_API_KEY",
        ModelClass.GEMINI
    )
    NOVA_PRO_V1 = (
        "NOVA_PRO_V1",
        "amazon.nova-pro-v1:0",
        "BEDROCK_API_KEY",
        ModelClass.NOVA,
        OtherArgs(json_tool=True, region="us-east-1")
    )
    MISTRAL_LARGE_2411 = (
        "MISTRAL_LARGE_2411",
        "mistral-large-2411",
        "MISTRAL_API_KEY",
        ModelClass.MISTRAL
    )
    
    def get_model_instance(
        self,
        config: str = "base",
        print_response: bool = False
    ):
        model_class, model_configs = self.model_class.impl
        config_json = validate_json(
            CONFIGS[config][self.key],
            model_configs
        )
        return model_class(
            api_key=get_env_var(self.api_key),
            model=self.model_name,
            configs=config_json,
            print_response=print_response,
            json_tool=self.other_args.json_tool,
            region=self.other_args.region
        )