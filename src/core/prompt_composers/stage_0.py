"""
Contains the Stage0PromptComposer model.
"""
import logging
from typing import List, Dict

from helpers.utils import file_to_string
from core.prompt_composers.base_composer import BaseComposer
from _types.prompts import Prompts
from _types.test_output import TestOutput


log = logging.getLogger("app")



class Stage0PromptComposer(BaseComposer):
    """
    Stage0PromptComposer model.
    
    Used for composing the prompts used in stage 0.
    """
    def _task_overview(self):
        return super()._task_overview()
    
    def _experimental_context(self):
        return super()._experimental_context()
    
    def _summary_context(self):
        # No summary context, as one would expect.
        return ""
    
    def _task(self, case: str):
        section_directory = self.sections_path / "task" / f"stage_{self.stage_name}"
        section_path = section_directory / f"{case}.md"
        return self._get_section(section_path, "Task")
    
    def _constraints(self):
        return super()._constraints()
    
    def _data_definitions(self, case: str):
        section = super()._data_definitions()
        
        section_path = self.sections_path / "data_definitions"
        stage_path = section_path / f"stage_{self.stage_name}"
        
        # Raw data definitions.
        section += file_to_string(stage_path / f"data_{self.treatment}.md")
        
        # Instance type definition.
        instance_type_path = stage_path / "instance_type"
        section += file_to_string(instance_type_path / "initial.md")
        section += file_to_string(instance_type_path / f"{case}.md")
        
        return section
    
    def _construct_system_prompt(self, case: str):
        a = self._task_overview()
        b = self._experimental_context()
        c = self._task(case)
        d = self._constraints()
        e = self._data_definitions(case)
        prompt = a + b + c + d + e
        return prompt
    
    def _construct_user_prompt(self, case: str) -> List[List[Dict]]:
        prompt = self.data.get_list_of_raw_instances(case)
        return prompt
    
    def expected_outputs(self):
        return self.data._len_windows()

    def get_prompts(self, test_outputs: List[TestOutput]):
        prompts = []
        
        # System prompt & user prompt dependent on case for stage 0.
        for test_output in test_outputs:
            n = test_output.replication
            for case in self.cases:
                system_prompt = self._construct_system_prompt(case)
                user_prompt = self._construct_user_prompt(case)
                
                # Prompts are iterative in stage 0 (ind. prompts for each summary).
                prompts.extend([
                    (
                        self._prompt_id("full", n, user),
                        Prompts(system=str(system_prompt), user=str(user))
                    )
                    for user in user_prompt
                ])
        
        return prompts
                
            
