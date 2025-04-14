"""
IRPD prompt module.

Contains the PromptComposer and Data models.
"""
import logging
import pandas as pd

from helpers.utils import file_to_string, to_list
from core.functions import categories_to_txt, output_attrb
from core.output_manager import OutputManager
from _types.prompts import Prompts


log = logging.getLogger(__name__)



class PromptComposer:
    """
    PromptComposer model.
    
    Gets the user and system prompts for a given stage, replication, and subset.
    """
    def __init__(self, output_manager: OutputManager):
        self.output_manager = output_manager
        
        # Categories are fixed for stages 2 & 3 if a 'replication' test type.
        self.fixed = self.irpd_config.test_type in {"cross_model", "intra_model"}
        
        self.data_path = self.irpd_config.data_path
        self.prompts_path = self.irpd_config.prompts_path
        self.sections_path = self.prompts_path / "sections"
        self.fixed_path = self.prompts_path / "fixed"
    
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
    
    def _data_definitions(self):
        """
        Returns the 'Data Variable Definitions' section of the system prompt.
        """
        section_path = self.sections_path / "data_definitions"
        stage_path = section_path / f"stage_{self.stage_name}"
        
        # Initial prompt section (header).
        section = file_to_string(section_path / "initial.md")
        
        # RA summary context.
        if self.stage_name not in {"0"}:
            section += file_to_string(stage_path / f"{self.ra}.md")
        
        # Instance type definition.
        subset_path = stage_path / "instance_type"
        section += file_to_string(subset_path / "initial.md")
        for case in self.cases:
            section += file_to_string(subset_path / f"{case}.md")
        
        # Adding definition of category assignment variable.
        if self.stage_name in {"3"}:
            section += file_to_string(stage_path / f"assignment.md")
        
        # Window number definition.
        section += file_to_string(stage_path / "window_number.md")
        
        return section
    
    def _construct_system_prompt(self):
        """
        Constructs the system prompts & sets as the system attrb.
        """
        a = self._task_overview()
        b = self._experimental_context()
        c = self._summary_context()
        d = self._task()
        e = self._constraints()
        prompt = a + b + c + d + e
        
        # Adding 'Data Variable Definitions' to stages w/ data in user prompt.
        if self.stage_name in {"1", "2", "3"}:
            prompt += self._data_definitions()
        
        # Appending a 'Categories' section to classification stages.
        if self.stage_name in {"2", "3"}:
            prompt += "\n\n## Categories\n\n"
            
            # Almost always will use Stage 1c categories, but if skipped, this 
            # should adjust the appended categories to include all stage 1r 
            # subset categories.
            if "1c" in self.irpd_config.stages:
                context = self.output_manager.retrieve(
                    self.llm_str, self.replication, "1c", "full"
                )
            else:
                context = self.output_manager.retrieve(
                    self.llm_str, self.replication, "1r"
                )
            for output in context:
                categories = output_attrb(output.outputs[0].parsed)
                prompt += categories_to_txt(categories)
        
        self.system = prompt
        return None
    
    def _construct_user_prompt(self):
        """
        Sets the user attrb. as (all) user prompt(s).
        """
        # Getting the correct RA summary data if a data dependent stage.
        if "0" not in self.irpd_config.stages and self.stage not in {"1", "1r"}:
            # Loading RA summary data.
            summary_path = self.data_path / "ra_summaries.csv"
            df = pd.read_csv(summary_path)
            
            # Getting all summary data in cases attrb. (in case of case 
            # composition).
            df = df[(df["case"].isin(self.cases))]
            
            # Adjusting for the treatment (if not merged).
            if self.treatment != "merged":
                df = df[(df["treatment"] == self.treatment)]
            
            # Adjusting for a singular RA.
            if self.ra != "both":
                ra_cols = [
                    c for c in df.columns
                    if c.startswith("summary_") and c != f"summary_{self.ra}"
                ]
                df = df.drop(columns=ra_cols)
            
            # Dropping unused variables.
            df = df.drop(columns=["case", "treatment", "subset"])
        else:
            log.error("Stage 0 has not been setup yet for prompts.")
            raise ValueError
        
        # Stage 1 is all summary data (in records form).
        if self.stage_name == "1":
            self.user = [df.to_dict("records")] # Essentially nested list.
        
        # Stage 1r & 1c user prompt is the categories created in prior stage.
        if self.stage_name in {"1r", "1c"}:
            # Want all subsets if stage 1c.
            subset = self.subset if self.stage_name == "1r" else None
            
            # Stage 1r uses stage 1 categories, stage 1c uses stage 1r categories.
            context_stage = "1" if self.stage_name == "1r" else "1r"
            
            context = self.output_manager.retrieve(
                self.llm_str, self.replication, context_stage, subset
            )
            
            prompt = ""
            for output in context:
                categories = output_attrb(output.outputs[0].parsed)
                prompt += categories_to_txt(categories)
            self.user = to_list(prompt)
        
        # Individual summaries for stages 2 & 3.
        if self.stage_name in {"2", "3"}:
            current_outputs = self.output_manager.retrieve(
                self.llm_str, self.replication, self.stage_name, self.subset
            )
            
            # Adjusting for max instances if specified.
            if self.irpd_config.max_instances:
                df = df[:self.irpd_config.max_instances]
            
            # Adjusting for completed classifications in output. Really only 
            # adjusted if test wasn't complete or failed before.
            if current_outputs[0].outputs:
                window_nums = [
                    output.parsed.window_number 
                    for output in current_outputs.outputs
                ]
                df = df[~df["window_number"].isin(window_nums)]
            
            # Stage 3 adds another variable for the classifications in stage 2.
            if self.stage_name in {"3"}:
                stage_2 = self.output_manager.retrieve(
                    self.llm_str, self.replication, "2", self.subset
                )
                for output in stage_2[0].outputs:
                    assigned_cats = [
                        cat.category_name
                        for cat in output.parsed.assigned_categories
                    ]
                    
                    window_number = output.parsed.window_number
                    df_index = df[df["window_number"] == window_number].index
                    
                    df.loc[df_index, "assigned_categories"] = str(assigned_cats)
            self.user = df.to_dict("records")
        
        return None
    
    def get_prompts(self):
        """
        Returns a list of all prompts.
        
        The total number of prompts should be the following:
            - Stage 0: Length of number of instances in case.
            - Stage 1: Length of the total number of subsets.
            - Stage 1r: Length of the total number of subsets.
            - Stage 1c: One prompt.
            - Stage 2: Length of number of summaries for case (adjusted for 
            already completed).
            - Stage 3: Length of number of summaries for case (adjusted for 
            already completed).
        """
        if self.fixed:
            return None
        self._construct_system_prompt()
        self._construct_user_prompt()
        return [Prompts(system=self.system, user=user) for user in self.user]
    


class Data:
    """
    Data model.
    
    Used to prepare data for prompts.
    """
    pass