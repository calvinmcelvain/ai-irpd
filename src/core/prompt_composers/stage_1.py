"""
Contains the Stage1PromptComposer model.
"""
import logging
from typing import List, Dict

from helpers.utils import file_to_string, to_list, load_config
from core.functions import categories_to_txt, output_attrb
from core.prompt_composers.base_composer import BaseComposer
from core.data import Data
from _types.prompts import Prompts
from _types.irpd_config import IRPDConfig
from _types.test_output import TestOutput


log = logging.getLogger("app")



class Stage1PromptComposer(BaseComposer):
    """
    Stage1PromptComposer model.
    
    Composes the prompts for stage 1.
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
        section =  super()._data_definitions()
        
        section_path = self.sections_path / "data_definitions"
        stage_path = section_path / f"stage_{self.stage_name}"
        
        # Summary variable context.
        section += file_to_string(stage_path / f"{self.ra}.md")
        
        # Instance type definition.
        instance_type_path = stage_path / "instance_type"
        section += file_to_string(instance_type_path / "initial.md")
        for case in self.cases:
            section += file_to_string(instance_type_path / f"{case}.md")
        
        # Window number definition.
        section += file_to_string(stage_path / "window_number.md")
        
        return section
    
    def _construct_system_prompt(self):
        a = self._task_overview()
        b = self._experimental_context()
        c = self._summary_context()
        d = self._task()
        e = self._constraints()
        f = self._data_definitions()
        prompt = a + b + c + d + e + f
        return prompt

    def _construct_user_prompt(
        self, test_output: TestOutput, subset: str
    ) -> List[Dict]:
        # If LLM summaries, use the CSV from stage 0 output.
        sub_path = test_output.sub_path if self.ra == "llm" else None
        prompt = [
            self.data.filter_summary_data(subset, sub_path).to_dict("records")
        ]
        return prompt
    
    def expected_outputs(self):
        return len(self.subsets[self.stage_name])
    
    def get_prompts(self, test_outputs: List[TestOutput]):
        prompts = []
        
        for test_output in test_outputs:
            n = test_output.replication
            system_prompt = self._construct_system_prompt()
            
            for subset in self.subsets[self.stage_name]:
                user_prompts = self._construct_user_prompt(test_output, subset)
                
                prompts.extend([
                    (
                        self._prompt_id(subset, n, user),
                        Prompts(system=str(system_prompt), user=str(user))
                    )
                    for user in user_prompts
                ])
        
        return prompts

