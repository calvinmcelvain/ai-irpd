import logging
import pandas as pd
from pydantic import BaseModel

from utils import file_to_string
from models.prompts import Prompts
from models.irpd.test_configs import StageConfig
from models.irpd.managers import OutputManager


log = logging.getLogger(__name__)



class TestPrompts:
    def __init__(
        self,
        stage_config: StageConfig,
        output_manager: OutputManager
    ):
        self.stage = stage_config.stage_name
        self.replication = stage_config.replication
        self.subset = stage_config.subset
        self.llm_str = stage_config.llm_str
        self.case = stage_config.case
        self.cases = stage_config.cases
        self.treatment = stage_config.treatment
        self.ra = stage_config.ra
        self.stage_config = stage_config
        self.output_manager = output_manager
        
        self.fixed = stage_config.test_type in {"cross_model", "intra_model"}
        
        self.data_path = stage_config.data_path
        self.prompts_path = stage_config.prompts_path
        self.sections_path = self.prompts_path / "sections"
        self.fixed_path = self.prompts_path / "fixed"
        
        self._construct_system_prompt()
        self._construct_user_prompt()
        
        self.expected_outputs = len(self.user)
    
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
            for case in self.cases:
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
                if "1c" in self.stage_config.stages:
                    context = self.output_manager.retrieve(
                        self.llm_str, self.replication, "1c", "full"
                    )
                else:
                    context = self.output_manager.retrieve(
                        self.llm_str, self.replication, "1r"
                    )
                for output in context:
                    categories = self._get_att(output.outputs[0].parsed)
                    prompt += self._categories_to_txt(categories)
        self.user = prompt
        return None
    
    def _construct_user_prompt(self):
        summary_path = self.data_path / "ra_summaries.csv"
        if "0" not in self.stage_config.stages:
            df = pd.read_csv(summary_path)
            df = df[(df["case"].isin(self.cases))]
            if self.treatment != "merged":
                df = df[(df["treatment"] == self.treatment)]
            if self.ra != "both":
                ra_cols = [
                    c for c in df.columns
                    if c.startswith("summary_") and c != f"summary_{self.ra}"
                ]
                df = df.drop(columns=ra_cols)
            if self.subset != "full":
                df = df.drop(columns=["instance_type"])
                df = df[(df["subset"] == self.subset)]
            df = df[(df["case"].isin(self.cases))]
            df = df.drop(columns=["case", "treatment", "subset"])
        else:
            log.error("Stage 0 has not been setup yet for prompts.")
            raise ValueError
        if self.stage == "1":
            self.user = [df.to_dict("records")]
        if self.stage == "1r":
            context = self.output_manager.retrieve(
                self.llm_str, self.replication, "1", self.subset
            )
            categories = context[0].outputs[0].parsed
            self.user = [self._categories_to_txt(self._get_att(categories))]
        if self.stage == "1c":
            prompt = ""
            context = self.output_manager.retrieve(self.llm, self.replication, "1r")
            for output in context:
                categories = self._get_att(output.outputs[0].parsed)
                prompt += self._categories_to_txt(categories)
            self.user = [prompt]
        if self.stage in {"2", "3"}:
            current_outputs = self.output_manager.retrieve(
                self.llm_str, self.replication, self.stage, self.subset
            )
            if self.stage_config.max_instances:
                df = df[:self.stage_config.max_instances]
            if self.stage in current_outputs.outputs:
                window_nums = [output.parsed.window_number for output in current_outputs.outputs]
                df = df[~df["window_number"].isin(window_nums)]
            if self.stage in {"3"}:
                stage_2 = self.output_manager.retrieve(
                    self.llm_str, self.replication, "2", self.subset
                )
                for output in stage_2.outputs:
                    assigned_cats = [cat.category_name for cat in output.parsed.assigned_categories]
                    df["assigned_categories"] = str(assigned_cats)
            self.user = df.to_dict("records")
        return None
    
    def get_prompts(self):
        if self.fixed:
            return None
        self._construct_system_prompt()
        self._construct_user_prompt()
        return [Prompts(system=self.system, user=user) for user in self.user]