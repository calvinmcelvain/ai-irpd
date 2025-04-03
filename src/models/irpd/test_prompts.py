import logging
import pandas as pd
from pydantic import BaseModel

from utils import get_env_var, file_to_string, str_to_path
from models.prompts import Prompts
from models.irpd.test_configs import TestConfig
from models.irpd.outputs import TestOutput


log = logging.getLogger(__name__)



class TestPrompts:
    def __init__(
        self,
        stage: str,
        replication: int,
        subset: str,
        llm: str,
        test_config: TestConfig,
        context: TestOutput
    ):
        self.stage = stage
        self.replication = replication
        self.subset = subset
        self.llm = llm
        self.case = test_config.case
        self.treatment = test_config.treatment
        self.ra = test_config.ra
        self.config = test_config
        self.context = context
        
        self.fixed = test_config.test_type in {"cross_model", "intra_model"}
        
        self.data_path = str_to_path(get_env_var("DATA_PATH"))
        self.prompts_path = str_to_path(get_env_var("PROMPTS_PATH"))
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
    
    def _data_definitions(self):
        section_path = self.sections_path / "data_definitions"
        stage_path = section_path / f"stage_{self.stage}"
        
        section = file_to_string(section_path / "initial.md")
        
        section += file_to_string(stage_path / f"{self.ra}.md")
        if self.subset == "full":
            subset_path = stage_path / "instance_type"
            section += file_to_string(subset_path / "initial.md")
            for case in self.case.split("_"):
                section += file_to_string(subset_path / f"{case}.md")
        if self.stage in {"3"}:
            section += file_to_string(stage_path / f"assignment.md")
        
        section += file_to_string(stage_path / "window_number.md")
        
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
    
    def _construct_system_prompt(self):
        a = self._task_overview()
        b = self._experimental_context()
        c = self._summary_context()
        d = self._task()
        e = self._constraints()
        prompt = a + b + c + d + e
        
        if self.stage in {"1", "2", "3"}:
            prompt += self._data_definitions()
            if self.stage in {"2", "3"}:
                prompt += "\n\n## Categories\n\n"
                context = self.context.retrieve("1c", self.llm, self.replication, "full")
                if context:
                    categories = self._get_att(context[0].parsed)
                    prompt += self._categories_to_txt(categories)
                else:
                    context = self.context.retrieve("1r", self.llm, self.replication)
                    for k in context.keys():
                        categories = self._get_att(context[k][0].parsed)
                        prompt += self._categories_to_txt(categories)
        self.user = prompt
        return None
    
    def _construct_user_prompt(self):
        summary_path = self.data_path / "ra_summaries.csv"
        cases = [c for c in self.case.split("_")]
        if {"0"} not in self.config.stages:
            df = pd.read_csv(summary_path)
            df = df[(df["case"].isin(cases))]
            if self.treatment != "merged":
                df = df[(df["treatment"] == self.treatment)]
            if self.ra != "both":
                ra_cols = [c for c in df.columns if c.startswith("summary_") and c != f"summary_{self.ra}"]
                df = df.drop(columns=ra_cols)
            if self.subset != "full":
                df = df.drop(columns=["instance_type"])
                df = df[(df["subset"] == self.subset)]
            df = df[(df["case"].isin(cases))]
            df = df.drop(columns=["case", "treatment", "subset"])
        else:
            log.error("Stage 0 has not been setup yet for prompts.")
            raise ValueError
        if self.stage == "1":
            self.user = [df.to_dict("records")]
        if self.stage == "1r":
            context = self.context.retrieve("1", self.llm, self.replication, self.subset)
            categories = context[0].parsed
            self.user = [self._categories_to_txt(self._get_att(categories))]
        if self.stage == "1c":
            prompt = ""
            context = self.context.retrieve("1r", self.llm, self.replication)
            for k in context.keys():
                categories = self._get_att(context[k][0].parsed)
                prompt += self._categories_to_txt(categories)
            self.user = [prompt]
        if self.stage in {"2", "3"}:
            if self.config.max_instances:
                df = df[:self.config.max_instances]
            if self.stage in self.context.stage_outputs.keys():
                context = self.context.retrieve(self.stage, self.llm, self.replication, self.subset)
                window_nums = [r.parsed.window_number for r in context]
                df = df[~df["window_number"].isin(window_nums)]
            if self.stage in {"3"}:
                stage_2 = self.context.retrieve("2", self.llm, self.replication, self.subset)
                for r in stage_2:
                    assigned_cats = [c.category_name for c in r.parsed.assigned_categories]
                    df["assigned_categories"] = str(assigned_cats)
            self.user = df.to_dict("records")
        return None
            
    def get_prompts(self) -> Prompts:
        if self.fixed:
            return None
        self._construct_system_prompt()
        self._construct_user_prompt()
        return [Prompts(system=self.system, user=user) for user in self.user]