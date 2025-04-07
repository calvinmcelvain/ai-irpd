import logging
import pandas as pd

from tools.functions import categories_to_txt, output_attrb
from utils import file_to_string
from models.prompts import Prompts
from models.irpd.output_manager import OutputManager


log = logging.getLogger(__name__)



class TestPrompts:
    def __init__(
        self,
        llm_str: str,
        stage_name: str,
        replication: int,
        subset: str,
        output_manager: OutputManager
    ):
        self.stage_name = stage_name
        self.replication = replication
        self.subset = subset
        self.llm_str = llm_str
        self.test_config = output_manager.test_config
        self.case = self.test_config.case
        self.cases = self.test_config.cases
        self.treatment = self.test_config.treatment
        self.ra = self.test_config.ra
        self.output_manager = output_manager
        
        self.fixed = self.test_config.test_type in {"cross_model", "intra_model"}
        
        self.data_path = self.test_config.data_path
        self.prompts_path = self.test_config.prompts_path
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
        section_path = self.sections_path / "task_overview" / f"stage_{self.stage_name}.md"
        return self._get_section(section_path, "Task Overview")

    def _experimental_context(self):
        section_path = self.sections_path / "experimental_context" / f"{self.treatment}.md"
        return self._get_section(section_path, "Experimental Context")
    
    def _summary_context(self):
        section_path = self.sections_path / "summary_context" / f"{self.case}_{self.ra}.md"
        return self._get_section(section_path, "Summary Context")
    
    def _task(self):
        section_path = self.sections_path / "task" / f"stage_{self.stage_name}.md"
        return self._get_section(section_path, "Task")
    
    def _constraints(self):
        section_path = self.sections_path / "constraints" / f"stage_{self.stage_name}.md"
        return self._get_section(section_path, "Constraints")
    
    def _data_definitions(self):
        section_path = self.sections_path / "data_definitions"
        stage_path = section_path / f"stage_{self.stage_name}"
        
        section = file_to_string(section_path / "initial.md")
        
        section += file_to_string(stage_path / f"{self.ra}.md")
        if self.subset == "full":
            subset_path = stage_path / "instance_type"
            section += file_to_string(subset_path / "initial.md")
            for case in self.cases:
                section += file_to_string(subset_path / f"{case}.md")
        if self.stage_name in {"3"}:
            section += file_to_string(stage_path / f"assignment.md")
        
        section += file_to_string(stage_path / "window_number.md")
        
        if not section:
            log.warning(f"PROMPTS: Data Definitions was empty.")
            return section
        return section
    
    def _construct_system_prompt(self):
        a = self._task_overview()
        b = self._experimental_context()
        c = self._summary_context()
        d = self._task()
        e = self._constraints()
        prompt = a + b + c + d + e
        
        if self.stage_name in {"1", "2", "3"}:
            prompt += self._data_definitions()
            if self.stage_name in {"2", "3"}:
                prompt += "\n\n## Categories\n\n"
                if "1c" in self.test_config.stages:
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
        summary_path = self.data_path / "ra_summaries.csv"
        if "0" not in self.test_config.stages:
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
        if self.stage_name == "1":
            self.user = [df.to_dict("records")]
        if self.stage_name == "1r":
            context = self.output_manager.retrieve(
                self.llm_str, self.replication, "1", self.subset
            )
            categories = context[0].outputs[0].parsed
            self.user = [categories_to_txt(output_attrb(categories))]
        if self.stage_name == "1c":
            prompt = ""
            context = self.output_manager.retrieve(self.llm_str, self.replication, "1r")
            for output in context:
                categories = output_attrb(output.outputs[0].parsed)
                prompt += categories_to_txt(categories)
            self.user = [prompt]
        if self.stage_name in {"2", "3"}:
            current_outputs = self.output_manager.retrieve(
                self.llm_str, self.replication, self.stage_name, self.subset
            )[0]
            if self.test_config.max_instances:
                df = df[:self.test_config.max_instances]
            if current_outputs.outputs:
                window_nums = [output.parsed.window_number for output in current_outputs.outputs]
                df = df[~df["window_number"].isin(window_nums)]
            if self.stage_name in {"3"}:
                stage_2 = self.output_manager.retrieve(
                    self.llm_str, self.replication, "2", self.subset
                )
                for output in stage_2[0].outputs:
                    assigned_cats = [cat.category_name for cat in output.parsed.assigned_categories]
                    df_index = df[df["window_number"] == output.parsed.window_number].index
                    df.loc[df_index, "assigned_categories"] = str(assigned_cats)
            self.user = df.to_dict("records")
        return None
    
    def get_prompts(self):
        if self.fixed:
            return None
        self._construct_system_prompt()
        self._construct_user_prompt()
        return [Prompts(system=self.system, user=user) for user in self.user]