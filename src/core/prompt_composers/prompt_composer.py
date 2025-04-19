"""
Contains the PromptComposer model.

Aggregates all PromptComposer models.
"""
from dataclasses import dataclass
from typing import TypeVar, Type
from enum import Enum
from functools import cached_property

from helpers.utils import dynamic_import
from core.prompt_composers.base_composer import BaseComposer
from _types.irpd_config import IRPDConfig


T = TypeVar("T", bound=BaseComposer)



@dataclass(frozen=True)
class PromptComposerContainer:
    module: str
    test_class: str
    stage_name: str
    
    @cached_property
    def impl(self) -> Type[T]:
        model_class: Type[T] = dynamic_import(self.module, self.test_class)
        return model_class



class PromptComposer(PromptComposerContainer, Enum):
    STAGE_0 = ("core.prompt_composers.stage_0", "Stage0PromptComposer", "0")
    STAGE_1 = ("core.prompt_composers.stage_1", "Stage1PromptComposer", "1")
    STAGE_1r = ("core.prompt_composers.stage_1r", "Stage1rPromptComposer", "1r")
    STAGE_1c = ("core.prompt_composers.stage_1c", "Stage1cPromptComposer", "1c")
    STAGE_2 = ("core.prompt_composers.stage_2", "Stage2PromptComposer", "2")
    STAGE_3 = ("core.prompt_composers.stage_3", "Stage3PromptComposer", "3")
    
    def get_prompt_composer(self, irpd_config: IRPDConfig):
        """
        Gets the prompt composer instance.
        """
        return self.impl(irpd_config=irpd_config, stage_name=self.stage_name)
    
