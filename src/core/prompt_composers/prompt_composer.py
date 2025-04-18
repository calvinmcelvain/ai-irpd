"""
Contains the PromptComposer model.

Aggregates all PromptComposer models.
"""
from dataclasses import dataclass
from enum import Enum
from functools import cached_property

from helpers.utils import dynamic_import
from _types.irpd_config import IRPDConfig



@dataclass(frozen=True)
class PromptComposerContainer:
    module: str
    test_class: str
    
    @cached_property
    def impl(self):
        return dynamic_import(self.module, self.test_class)



class PromptComposer(PromptComposerContainer, Enum):
    stage_0 = ("core.prompt_composers.stage_0", "Stage0PromptComposer")
    stage_1 = ("core.prompt_composers.stage_1", "Stage1PromptComposer")
    stage_1r = ("core.prompt_composers.stage_1r", "Stage1rPromptComposer")
    stage_1c = ("core.prompt_composers.stage_1c", "Stage1cPromptComposer")
    stage_2 = ("core.prompt_composers.stage_2", "Stage2PromptComposer")
    stage_3 = ("core.prompt_composers.stage_3", "Stage3PromptComposer")
    
    def get_prompt_composer(self, irpd_config: IRPDConfig):
        """
        Gets the prompt composer instance.
        """
        return self.impl(irpd_config=irpd_config)
    
