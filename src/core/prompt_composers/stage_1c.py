"""
Contains the Stage1cPromptComposer model.
"""
import logging
from typing import List

from core.functions import output_attrb, categories_to_txt
from core.prompt_composers.base_composer import BaseComposer
from _types.prompts import Prompts
from _types.test_output import TestOutput


log = logging.getLogger("app")



class Stage1cPromptComposer(BaseComposer):
    """
    Stage1cPromptComposer model.
    
    Composes the prompts for stage 1c.
    """
    def _task_overview(self):
        return super()._task_overview()
    
    def _experimental_context(self):
        return super()._experimental_context()
    
    def _summary_context(self):
        if "llm" in self.ra:
            return None
        return super()._summary_context()
    
    def _task(self):
        return super()._task()
    
    def _constraints(self):
        return super()._constraints()
    
    def _data_definitions(self):
        # Not data dependent stage
        return ""
    
    def _construct_system_prompt(self):
        a = self._task_overview()
        b = self._experimental_context()
        c = self._summary_context()
        d = self._task()
        e = self._constraints()
        prompt = a + b + c + d + e
        return prompt

    def _construct_user_prompt(self, test_output: TestOutput) -> str:
        # User prompt is all subset categories created from stage 1r.
        context = test_output.stage_outputs["1r"].outputs
        prompt = ""
        for output in context.values():
            categories = output_attrb(output[0].parsed)
            prompt += categories_to_txt(categories)
        return prompt
    
    def expected_outputs(self):
        return 1
    
    def get_prompts(self, test_outputs: List[TestOutput]):
        prompts = []
        
        for test_output in test_outputs:
            n = test_output.replication
            system_prompt = self._construct_system_prompt()
            
            user_prompt = self._construct_user_prompt(test_output)
            
            prompts.append((
                self._prompt_id("full", n, user_prompt),
                Prompts(system=str(system_prompt), user=str(user_prompt))
            ))
        
        return prompts

