import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product
from abc import ABC, abstractmethod

from models.irpd.test_config import TestConfig
from models.irpd.test_output import TestOutput
from models.llms.base_llm import BaseLLM
from utils import validate_json_string, write_json


log = logging.getLogger(__name__)



class BaseStage(ABC):
    def __init__(
        self,
        test_config: TestConfig,
        sub_path: Path,
        context: TestOutput,
        llm: BaseLLM
    ):
        self.case = test_config.case
        self.cases = self.case.split("_")
        self.ra = test_config.ra
        self.treatment = test_config.treatment
        self.llm = llm
        self.test_path = test_config.test_path
        self.sub_path = sub_path
        self.context = context
        self.subsets = self._get_subsets()
    
    @staticmethod
    def _get_instance_types(case: str):
        if case in {"uni", "uniresp"}:
            return ["ucoop", "udef"]
        return ["coop", "def"]
    
    def _get_subsets(self):
        instance_types = [self._get_instance_types(c) for c in self.cases]
        return product(self.cases, instance_types)
    
    def _compute_tokens(self):
        tokens = {case: {} for case in self.cases}
        for c in self.cases:
            for i in self._get_instance_types(c):
                outputs = self.output.get(c, i)
                tokens[c][i] = {
                    "input_tokens": sum(output.meta.usage.prompt_tokens for output in outputs),
                    "output_tokens": sum(output.meta.usage.completion_tokens for output in outputs),
                    "total_tokens": sum(output.meta.usage.total_tokens for output in outputs)
                }
        tokens["total"] = {
            "input_tokens": sum(
                tokens[c][i]["input_tokens"] for c in self.cases for i in self._get_instance_types(c)),
            "output_tokens": sum(tokens[c][i]["output_tokens"] for c in self.cases for i in self._get_instance_types(c)),
            "total_tokens": sum(tokens[c][i]["total_tokens"] for c in self.cases for i in self._get_instance_types(c))
        }
        return tokens
    
    def _write_meta(self):
        model = self.llm.model
        parameters = self.llm.configs
        created = str(datetime.now())
        try:
            tokens = self._compute_tokens()
        except AttributeError:
            tokens = None
        
        meta_path = self.sub_path / "_test_info"
        meta_path.mkdir(exist_ok=True)
        
        json_data = {
            "created": created,
            "model_info": {
                "model": model,
                "parameters": parameters.model_dump()
            },
            "test_info": {
                "case": self.case,
                "ra": self.ra,
                "treatment": self.treatment,
                "threshold": self.threshold,
                "test_type": self.test_type,
                "test_path": self.test_path.relative_to(self.project_path).as_posix()
            },
            "tokens": tokens
        }
        
        path = meta_path / f"stg_{self.stage}_test_info.json"
        write_json(path, json_data)
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
    def _get_system_prompt(self):
        pass
    
    @abstractmethod
    def _get_user_prompt(self):
        pass
    
    @abstractmethod
    def _process_output(self):
        pass
    
    @abstractmethod
    def run(self):
        pass