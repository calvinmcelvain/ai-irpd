import logging
import pandas as pd
from pathlib import Path
from pydantic import BaseModel

from utils import txt_to_pdf
from models.irpd.test_meta import TestMeta
from models.irpd.outputs import TestOutput
from models.irpd.test_configs import TestConfig


log = logging.getLogger(__name__)



class OutputProcesser:
    def __init__(self, test_config: TestConfig):
        self.case = test_config.case
        self.cases = test_config.case.split("_")
        self.stages = test_config.stages
        self.test_path = test_config.test_path
        self.total_replications = test_config.total_replications
        self.llms = test_config.llms
        self.llm_config = test_config.llm_config
    
    @staticmethod
    def _get_instance_types(case: str):
        if case in {"uni", "uniresp"}:
            return ["ucoop", "udef"]
        return ["coop", "def"]
    
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
    
    def _generate_subpath(self, N: int, llm_str: str):
        subpath = self.test_path
        if len(self.llms) > 1: subpath = subpath / llm_str
        if self.total_replications > 1: subpath = subpath / f"replication_{N}"
        return Path(subpath)
    
    def _build_categories_pdf(self, stage: str):
        pdf = f"# Stage {stage} Categories\n\n"
        for subset in self.subsets:
            if subset in self.output.outputs.keys():
                output = self.output.outputs[subset][0]
                categories = self._get_att(output.parsed)
                if subset != "full":
                    case, sub = subset.split("_")
                    pdf += f"## {case.capitalize()}; {sub.upper()} Categories\n\n"
                else:
                    if stage == "1c":
                        pdf += f"## Final Category Set\n\n"
                    else:
                        pdf += f"## Unified Categories\n\n"
                pdf += self._categories_to_txt(categories=categories)
        pdf_path = self.sub_path / f"_stage_{stage}_categories.pdf"
        txt_to_pdf(text=pdf, file_path=pdf_path)
        return None
    
    def _build_data_output(self, stage: str):
        dfs = []
        for case in self.cases:
            raw_df_path = self.data_path / "raw" / f"{case}_{self.treatment}_{self.ra}.csv"
            raw_df = pd.read_csv(raw_df_path)
            df_list = []
            for subset in self.subsets:
                response_list = []
                outputs = self.output.outputs[subset]
                for output in outputs:
                    response = {}
                    response["reasoning"] = output.parsed.reasoning
                    response["window_number"] = output.parsed.window_number
                    for l in self._get_att(output.parsed):
                        response[l.category_name] = 1
                        if hasattr(l, "rank"):
                            response[l.category_name] = l.rank
                    response_list.append(response)
                response_df = pd.DataFrame.from_records(response_list)
                df_list.append(response_df)
            df = pd.concat(df_list, ignore_index=True, sort=False).fillna(0)
            merged_df = pd.merge(raw_df, df, on='window_number')
            merged_df["case"] = case
            dfs.append(merged_df)
        df = pd.concat(dfs, ignore_index=True, sort=False)
        df.to_csv(self.sub_path / f"_stage_{self.stage}_final.csv", index=False)
        return None
    
    def process(self, output: TestOutput):
        pass