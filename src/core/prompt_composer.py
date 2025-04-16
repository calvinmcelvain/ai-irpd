"""
Contains the PromptComposer.
"""
import logging
from typing import List, Tuple

from helpers.utils import file_to_string, to_list
from core.functions import categories_to_txt, output_attrb
from core.foundation import FoundationalModel
from core.data import Data
from _types.prompts import Prompts
from _types.irpd_config import IRPDConfig
from _types.test_output import TestOutput


log = logging.getLogger(__name__)



class PromptComposer(FoundationalModel):
    """
    PromptComposer model.
    
    Gets the user and system prompts for a given stage, replication, and subset.
    """
    def __init__(self, irpd_config: IRPDConfig):
        super().__init__(self, irpd_config)
        
        self.data = Data(irpd_config)
        
        # Categories are fixed for stages 2 & 3 if a 'replication' test type.
        self.fixed = self.irpd_config.test_type in {"cross_model", "intra_model"}

        self.sections_path = self.prompts_path / "sections"
        self.fixed_path = self.prompts_path / "fixed"
        
        self.expected_outputs = {
            stage: self._expected_outputs(stage)
            for stage in self.stages
        }
    
    @staticmethod
    def _get_section(section_path, name):
        """
        Returns the prompt section & logs if was found to be empty.
        """
        section = file_to_string(section_path)
        if not section:
            log.warning(f"{name} was empty.")
        return section + "\n"
    
    def _task_overview(self):
        """
        Returns the 'Task Overview' section of system prompt.
        """
        section_path = self.sections_path / "task_overview" / f"stage_{self.stage_name}.md"
        return self._get_section(section_path, "Task Overview")

    def _experimental_context(self):
        """
        Returns the 'Experimental Context' section of the system prompt.
        """
        section_path = self.sections_path / "experimental_context" / f"{self.treatment}.md"
        return self._get_section(section_path, "Experimental Context")
    
    def _summary_context(self):
        """
        Returns the 'Summary Context' section of the system prompt.
        """
        section_path = self.sections_path / "summary_context" / f"{self.case}_{self.ra}.md"
        return self._get_section(section_path, "Summary Context")
    
    def _task(self):
        """
        Returns the stasge task section(s) of the system prompt.
        
        Essentially just the instructions for the stage.
        """
        section_path = self.sections_path / "task" / f"stage_{self.stage_name}.md"
        return self._get_section(section_path, "Task")
    
    def _constraints(self):
        """
        Returns the 'Constraints' section of the system prompt.
        
        Apart of the stage specific instructions.
        """
        section_path = self.sections_path / "constraints" / f"stage_{self.stage_name}.md"
        return self._get_section(section_path, "Constraints")
    
    @staticmethod
    def _prompt_id(stage: str, subset: str, n: int, user: object):
        """
        Generates the prompt ID.
        """
        prompt_id = f"{n}-{subset}"
        if stage in {"2", "3"}:
            prompt_id += f"-{user["window_number"]}"
        return prompt_id
    
    def _data_definitions(self, stage_name: str):
        """
        Returns the 'Data Variable Definitions' section of the system prompt.
        """
        if stage_name not in {"1", "2", "3"}: return ""
        section_path = self.sections_path / "data_definitions"
        stage_path = section_path / f"stage_{stage_name}"
        
        # Initial prompt section (header).
        section = file_to_string(section_path / "initial.md")
        
        # RA summary context.
        if stage_name not in {"0"}:
            section += file_to_string(stage_path / f"{self.ra}.md")
        
        # Instance type definition.
        instance_type_path = stage_path / "instance_type"
        section += file_to_string(instance_type_path / "initial.md")
        for case in self.cases:
            section += file_to_string(instance_type_path / f"{case}.md")
        
        # Adding definition of category assignment variable.
        if stage_name in {"3"}:
            section += file_to_string(stage_path / f"assignment.md")
        
        # Window number definition.
        section += file_to_string(stage_path / "window_number.md")
        
        return section
    
    def _construct_system_prompt(self, test_output: TestOutput, stage_name: str):
        """
        Constructs the system prompt.
        """
        a = self._task_overview()
        b = self._experimental_context()
        c = self._summary_context()
        d = self._task()
        e = self._constraints()
        f = self._data_definitions(stage_name)
        
        prompt = a + b + c + d + e + f
        # Appending a 'Categories' section to classification stages.
        if stage_name in {"2", "3"}:
            prompt += "\n\n## Categories\n\n"
            
            # Almost always will use Stage 1c categories, but if skipped, this 
            # should adjust the appended categories to include all stage 1r 
            # subset categories.
            if "1c" in self.stages:
                context = test_output.stage_outputs["1c"].outputs
            else:
                context = test_output.stage_outputs["1r"].outputs
            
            for output in context.values():
                request_out = output[0].request_out
                categories = output_attrb(request_out.parsed)
                prompt += categories_to_txt(categories)

        return prompt
    
    def _construct_user_prompt(
        self, test_output: TestOutput, stage_name: str, subset: str
    ):
        """
        Returns user prompt.
        """
        # Stage 1 is all summary data (in records form).
        if stage_name == "1":
            prompt = self.data.filter_ra_data(subset)
        
        # Stage 1r user prompt is the categories created in stage 1.
        if stage_name == "1r":
            context = test_output.stage_outputs["1"].outputs[subset]
            categories = output_attrb(context[0].request_out.parsed)
            prompt = categories_to_txt(categories)
        
        # Stage 1c user prompt is all subset categories created in stage 1r.
        if stage_name == "1c":
            context = test_output.stage_outputs["1r"].outputs
            prompt = ""
            for output in context.values():
                request_out = output[0].request_out
                categories = output_attrb(request_out.parsed)
                prompt += categories_to_txt(categories)
        
        # Individual summaries for stages 2 & 3.
        if stage_name in {"2", "3"}:
            df = self.data.adjust_for_completed_outputs(test_output, stage_name)
            
            # Stage 3 adds another variable for the classifications in stage 2.
            if stage_name == "3":
                stage_2_outputs = test_output.stage_outputs["2"].outputs["full"]
                for output in stage_2_outputs:
                    request_out = output.request_out.parsed
                    assigned_cats = [
                        cat.category_name
                        for cat in request_out.assigned_categories
                    ]
                    df_index = df[df["window_number"] == request_out.window_number].index
                    
                    df.loc[df_index, "assigned_categories"] = str(assigned_cats)
            prompt = df.to_dict("records")
                
        return to_list(prompt)

    def _expected_outputs(self, stage_name: str) -> int:
        """
        Checks the expected number of outputs via the count of user prompts for 
        a stage.
        """
        if stage_name in {"1", "1r", "1c"}:
            return len(self.subsets[stage_name])
        return len(self.data.ra_data.to_dict("records"))
    
    def get_prompts(
        self, test_outputs: List[TestOutput], stage_name: str
    ) -> List[Tuple[str, Prompts]]:
        """
        Returns a list of tuples w/ first element the id, and the second a 
        Prompts object.
        """
        if self.fixed:
            return None
        
        prompts = []
        
        for test_output in test_outputs:
            n = test_output.replication
            system_prompt = self._construct_system_prompt(test_output, stage_name)
            
            for subset in test_output.stage_outputs[stage_name].outputs.keys():
                
                user_prompts = self._construct_user_prompt(
                    test_output, stage_name, subset)
                
                prompts.extend([
                    (
                        self._prompt_id(stage_name, subset, n, user),
                        Prompts(system=system_prompt, user=user)
                    )
                    for user in user_prompts
                ])
        return prompts
