"""
Contains the Stage2PromptComposer model.
"""
import logging
from typing import List, Dict

from helpers.utils import file_to_string
from core.functions import output_attrb, categories_to_txt
from core.prompt_composers.base_composer import BaseComposer
from _types.prompts import Prompts
from _types.test_output import TestOutput


log = logging.getLogger("app")



class Stage2PromptComposer(BaseComposer):
    """
    Stage2PromptComposer model.
    
    Composes the prompts for stage 2.
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
        section = super()._data_definitions()
        
        section_path = self.sections_path / "data_definitions"
        stage_path = section_path / f"stage_{self.stage_name}"
        
        # Summary variable definition.
        section += file_to_string(stage_path / f"{self.ra}.md")
        
        # Instance type definition.
        instance_type_path = stage_path / "instance_type"
        section += file_to_string(instance_type_path / "initial.md")
        for case in self.cases:
            section += file_to_string(instance_type_path / f"{case}.md")
            
        # Window number definition.
        section += file_to_string(stage_path / "window_number.md")
        
        return section
    
    def _construct_system_prompt(self, test_output: TestOutput) -> List[List[Dict]]:
        a = self._task_overview()
        b = self._experimental_context()
        c = self._summary_context()
        d = self._task()
        e = self._constraints()
        f = self._data_definitions()
        
        prompt = a + b + c + d + e + f
        
        # Appending a 'Categories' section to classification stages.
        prompt += "\n\n## Categories\n\n"
        
        # Almost always will use Stage 1c categories, but if skipped, this 
        # should adjust the appended categories to include all stage 1r 
        # subset categories.
        if "1c" in self.stages:
            context = test_output.stage_outputs["1c"].outputs
        else:
            context = test_output.stage_outputs["1r"].outputs
        
        for output in context.values():
            categories = output_attrb(output[0].parsed)
            prompt += categories_to_txt(categories)

        return prompt

    def _construct_user_prompt(self, test_output: TestOutput) -> str:
        # Individual summaries for stage 2.
        df = self.data.adjust_for_completed_outputs(test_output, self.stage_name)
        return df.to_dict("records")
    
    def expected_outputs(self):
        return len(self.data.ra_data[:self.max_instances].to_dict("records"))
    
    def get_prompts(self, test_outputs: List[TestOutput]):
        prompts = []
        
        for test_output in test_outputs:
            n = test_output.replication
            system_prompt = self._construct_system_prompt(test_output)
                
            user_prompts = self._construct_user_prompt(test_output)
            
            prompts.extend([
                (
                    self._prompt_id("full", n, user),
                    Prompts(system=str(system_prompt), user=str(user))
                )
                for user in user_prompts
            ])
        
        return prompts

