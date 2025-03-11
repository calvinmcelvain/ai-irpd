import logging
import pandas as pd
from pydantic import BaseModel
from pathlib import Path
from abc import ABC, abstractmethod

from models.irpd.test_config import TestConfig
from models.irpd.test_prompts import TestPrompts
from models.irpd.test_output import TestOutput
from models.llms.base_llm import BaseLLM
from models.request_output import RequestOut
from utils import (
    validate_json_string, write_json, load_json, str_to_path, get_env_var,
    lazy_import
)


log = logging.getLogger(__name__)



class BaseStage(ABC):
    def __init__(
        self,
        test_config: TestConfig,
        sub_path: Path,
        prompts: TestPrompts,
        context: TestOutput,
        llm: BaseLLM,
        **kwargs
    ):
        self.config = test_config
        self.case = test_config.case
        self.cases = self.case.split("_")
        self.ra = test_config.ra
        self.treatment = test_config.treatment
        self.llm = llm
        self.test_path = test_config.test_path
        self.sub_path = sub_path
        self.prompts = prompts
        self.context = context
        self.subsets = self._get_subsets()
        self.output_path = str_to_path(get_env_var("OUTPUT_PATH"))
        self.fixed = kwargs.get("fixed", False)
    
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
    
    def _check_context(self, subset: str):
        if self.context:
            if self.stage in self.context.stage_outputs.keys():
                if subset in self.context.stage_outputs[self.stage].outputs:
                    if isinstance(self.context.stage_outputs[self.stage].outputs[subset], RequestOut):
                        return True
        return None
    
    def _get_subsets(self):
        subsets = [f"{c}_{i}" for c in self.cases for i in self._get_instance_types(c)]
        return subsets + ["full"]
    
    def _get_stage_schema(self):
        return lazy_import("models.irpd.schemas", f"Stage{self.stage}Schema")
    
    def _write_meta(self):
        meta_path = self.sub_path / "_test_meta.json"
        if meta_path.exists():
            json_data = load_json(meta_path)
            for subset in self.subsets:
                output_meta = self.output.outputs[subset].meta
                json_data[self.stage][subset]["input_tokens"] += output_meta.input_tokens
                json_data[self.stage][subset]["input_tokens"] += output_meta.input_tokens
                json_data[self.stage][subset]["total_tokens"] += output_meta.total_tokens
        else:
            model = self.llm.model
            parameters = self.llm.configs.model_dump()
            
            json_data = {
                "model_info": {
                    "model": model,
                    "parameters": parameters.model_dump()
                },
                "test_info": {
                    "case": self.case,
                    "ra": self.ra,
                    "treatment": self.treatment,
                    "test_type": self.test_type,
                    "test_path": self.test_path.relative_to(self.output_path).as_posix()
                },
                "stages": {}
            }
            for stage in self.config.stages:
                json_data["stages"][stage] = {}
                for subset in self.subsets:
                    json_data["stages"][stage][subset] = {}
                    json_data["stages"][stage][subset]["input_tokens"] = 0
                    json_data["stages"][stage][subset]["output_tokens"] = 0
                    json_data["stages"][stage][subset]["total_tokens"] = 0
        
        write_json(meta_path, json_data)
        return None
    
    def _build_data_output(self):
        dfs = []
        for case in self.cases:
            raw_df_path = self.data_path / "raw" / f"{case}_{self.treatment}_{self.ra}.csv"
            raw_df = pd.read_csv(raw_df_path)
            df_list = []
            for i in self._get_instance_types(case):
                response_list = []
                outputs = self.output.get(case, i)
                for j in outputs:
                    response = {}
                    output = validate_json_string(j.response, self.schema)
                    response["reasoning"] = output.reasoning
                    response["window_number"] = output.window_number
                    for l in self._get_category_att(output):
                        name = f"{i}_{l.category_name}"
                        response[name] = 1
                        if hasattr(l, "rank"):
                            response[name] = l.rank
                    response_list.append(response)
                response_df = pd.DataFrame.from_records(response_list)
                df_list.append(response_df)
            df = pd.concat(df_list, ignore_index=True, sort=False).fillna(0)
            merged_df = pd.merge(raw_df, df, on='window_number')
            if len(self.cases) > 1:
                merged_df["case"] = self.cases.index(case)
            dfs.append(merged_df)
        return pd.concat(dfs, ignore_index=True, sort=False)
    
    @abstractmethod
    def _process_output(self):
        pass
    
    @abstractmethod
    def run(self):
        pass