import logging
import pandas as pd
from pydantic import BaseModel
from typing import Union, Optional
from pathlib import Path

from utils import get_env_var, file_to_string, str_to_path
from models.prompts import Prompts
from models.irpd.test_config import TestConfig
from models.irpd.test_output import TestOutput


log = logging.getLogger(__name__)



class TestPrompts:
    def __init__(
        self,
        stage: str,
        test_config: TestConfig,
        context: Optional[TestOutput],
        prompt_path: Union[str, Path] = None,
        data_path: Union[str, Path] = None
    ):
        self.stage = stage
        self.case = test_config.case
        self.treatment = test_config.treatment
        self.ra = test_config.ra
        self.config = test_config
        self.context = context
        
        self.data_path = str_to_path(data_path or get_env_var("DATA_PATH"))
        self.prompts_path = str_to_path(prompt_path or get_env_var("PROMPTS_PATH"))
        self.sections_path = self.prompts_path / "sections"
        self.fixed_path = self.prompts_path / "fixed"
    
    @staticmethod
    def _get_section(section_path, name):
        section = file_to_string(section_path)
        if not section:
            log.warning(f"PROMPTS: {name} was empty.")
        return section + "\n"
    
    def _task_overview(self):
        section_path = self.sections_path / "task_overview" / f"stage_{self.stage}.md"
        return self._get_section(section_path, "Task Overview")

    def _experimental_context(self):
        section_path = self.sections_path / "experimental_context" / f"{self.treatment}.md"
        return self._get_section(section_path, "Experimental Context")
    
    def _summary_context(self):
        section_path = self.sections_path / "summary_context" / f"{self.case}_{self.ra}.md"
        return self._get_section(section_path, "Summary Context")
    
    def _task(self):
        section_path = self.sections_path / "task" / f"stage_{self.stage}.md"
        return self._get_section(section_path, "Task")
    
    def _constraints(self):
        section_path = self.sections_path / "constraints" / f"stage_{self.stage}.md"
        return self._get_section(section_path, "Constraints")
    
    def _data_definitions(self, subset: str):
        section_path = self.sections_path / "data_definitions"
        
        section = file_to_string(section_path / "initial.md")
        
        if self.stage in {"1"}:
            section += file_to_string(section_path / "stage_1" / f"{self.ra}.md")
            if subset == "full":
                subset_path = section_path / "stage_1" / "subset"
                section += file_to_string(subset_path / "initial.md")
                for case in self.case.split("_"):
                    section += file_to_string(subset_path / f"{case}.md")
        
        section += file_to_string(section_path / "stage_1" / "window_number.md")
        
        if not section:
            log.warning(f"PROMPTS: Data Definitions was empty.")
            return section
        return section
    
    @staticmethod
    def _get_att(output):
        if hasattr(output, "categories"):
            return output.categories
        if hasattr(output, "refined_categories"):
            return output.refined_categories
        if hasattr(output, "assigned_categories"):
            return output.assigned_categories
        if hasattr(output, "category_ranking"):
            return output.category_ranking
    
    @staticmethod
    def _categories_to_txt(categories: BaseModel):
        category_texts = []
        for category in categories:
            example_texts = []
            for idx, example in enumerate(category.examples, start=1):
                example_texts.append(
                    f"  {idx}. Window number: {example.window_number},"
                    f" Reasoning: {example.reasoning}"
                )
            category_text = (
                f"### {category.category_name}\n\n"
                f"**Definition**: {category.definition}\n\n"
                f"**Examples**:\n\n{"\n".join(example_texts)}\n\n"
            )
            category_texts.append(category_text)
        return "".join(category_texts)
    
    def _construct_system_prompt(self, subset: str):
        a = self._task_overview()
        b = self._experimental_context()
        c = self._summary_context()
        d = self._task()
        e = self._constraints()
        prompt = a + b + c + d + e
        
        if self.stage in {"1"}:
            prompt += self._data_definitions(subset=subset)
        if self.stage in {"2", "3"}:
            prompt += "\n## Categories"
            context = self.context.stage_outputs.get("1r").outputs
            for k in context.keys():
                categories = self._get_att(context.get(k)[0].parsed)
                prompt += self._categories_to_txt(categories)
        return prompt
    
    def _construct_user_prompt(self, subset: str, case: str):
        summary_path = self.data_path / "ra_summaries.csv"
        if {"0"} not in self.config.stages:
            df = pd.read_csv(summary_path)
            cases = [c for c in self.case.split("_")]
            df = df[(df["case"].isin(cases))]
            if self.treatment != "merged":
                df = df[(df["treatment"] == self.treatment)]
            if self.ra != "both":
                ra_cols = [c for c in df.columns if c.startswith("summary_") and c != f"summary_{self.ra}"]
                df = df.drop(columns=ra_cols)
            if subset != "full":
                df = df.drop(columns=["subset"])
                df = df[(df["instance_type"] == subset.split("_")[1])]
            cases = [c for c in case.split("_")]
            df = df[(df["case"].isin(cases))]
            df = df.drop(columns=["case", "treatment", "instance_type"])
        else:
            log.error("Stage 0 has not been setup yet for prompts.")
            raise ValueError
        if self.stage == "1":
            return df.to_dict("records")
        if self.stage == "1r":
            context = self.context.stage_outputs
            categories = context.get("1").outputs.get(subset)[0].parsed
            return self._categories_to_txt(self._get_att(categories))
        if self.stage == "1c":
            prompt = ""
            context = self.context.stage_outputs.get("1r").outputs
            for k in context.keys():
                categories = self._get_att(context.get(k)[0].parsed)
                prompt += self._categories_to_txt(categories)
            return prompt
        if self.stage in {"2", "3"}:
            if self.config.max_instances:
                df = df[:self.config.max_instances]
            if self.stage in self.context.stage_outputs.keys():
                context = self.context.stage_outputs.get(self.stage).outputs
                window_nums = [r.window_number for r in context.get(subset)]
                df = df[~df["window_number"].isin(window_nums)]
            if self.stage == "3":
                stage_2 = self.context.stage_outputs.get("2").outputs.get(subset)
                for r in stage_2:
                    assigned_cats = [c.category_name for c in r.parsed.assigned_categories]
                    df["assigned_categories"] = assigned_cats
            return df.to_dict("records")
            
    def get_prompts(self, subset: str, case: str, fixed: bool = False) -> Prompts:
        if fixed:
            return None
        system = self._construct_system_prompt(subset=subset)
        user = self._construct_user_prompt(subset=subset, case=case)
        return Prompts(system=system, user=user)
        
        