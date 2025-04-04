import logging
import pandas as pd
from typing import List
from pathlib import Path
from pydantic import BaseModel

from utils import txt_to_pdf
from tools.functions import categories_to_txt, instance_types, output_attrb
from models.irpd.test_meta import TestMeta
from models.irpd.outputs import SubOutput, StageOutput


log = logging.getLogger(__name__)



class OutputProcesser:
    def __init__(self, sub_output: SubOutput):
        self.output = sub_output
        self.configs = sub_output.sub_config
        self.sub_path = sub_output.sub_config.sub_path
        self.cases = sub_output.sub_config.cases
        self.treatment = sub_output.sub_config.treatment
        self.ra = sub_output.sub_config.ra
        self.data_path = sub_output.sub_config.data_path
        self.stages = sub_output.sub_config.stages
        self.llm_instance = sub_output.sub_config.llm_instance
        self.batch_id = sub
    
    def _build_categories_pdf(self, stage_name: str, stage_outputs: List[StageOutput]):
        pdf = f"# Stage {stage_name} Categories\n\n"
        for output in stage_outputs:
            categories = output_attrb(output.outputs[0].parsed)
            
            subset = output.stage_config.subset
            
            if subset != "full":
                case, instance_type = subset.split("_")
                pdf += f"## {case.capitalize()} - {instance_type.upper()} Categories\n\n"
            else:
                if stage_name == "1c":
                    pdf += f"## Final Category Set\n\n"
                else:
                    pdf += f"## Unified Categories\n\n"
            pdf += categories_to_txt(categories)
        pdf_path = self.sub_path / f"_stage_{stage_name}_categories.pdf"
        txt_to_pdf(pdf, pdf_path)
        return None
    
    def _build_data_output(self, stage_name: str, stage_outputs: List[StageOutput]):
        dfs = []
        for case in self.configs.cases:
            raw_path = self.data_path / "raw"
            raw_df_path = raw_path / f"{case}_{self.treatment}_{self.ra}.csv"
            raw_df = pd.read_csv(raw_df_path)
            df_list = []
            
            for outputs in stage_outputs:
                response_list = []
                for output in outputs.outputs:
                    response = {"reasoning": output.parsed.reasoning}
                    response["window_number"] = output.parsed.window_number
                    for cat in output_attrb(output):
                        response[cat.category_name] = 1
                        if hasattr(cat, "rank"):
                            response[cat.category_name] = cat.rank
                    response_list.append(response)
                    response_df = pd.DataFrame.from_records(response_list)
                    df_list.append(response_df)
            df = pd.concat(df_list, ignore_index=True, sort=False).fillna(0)
            merged_df = pd.merge(raw_df, df, on='window_number')
            merged_df["case"] = case
            dfs.append(merged_df)
        df = pd.concat(dfs, ignore_index=True, sort=False)
        df.to_csv(self.sub_path / f"_stage_{stage_name}_final.csv", index=False)
        return None
    
    def _stage_info(self, stage_name: str, stage_outputs: List[StageOutput]):
        last_subset = stage_outputs[len(stage_outputs) - 1].outputs
        last_output = last_subset[len(last_subset)].meta
        
        stage_tokens = {}
        
        subsets = list(set(output.subset for output in stage_outputs))
        for subset in subsets:
            subset_outputs: List[StageOutput] = list(
                filter(lambda output: output.subset == subset, stage_outputs)
            )
            
            outputs = []
            for output in subset_outputs:
                outputs.extend(output.outputs)
            
            input_tokens = sum(
                [output.meta.input_tokens for output in outputs if output.meta]
            )
            output_tokens = sum(
                [output.meta.output_tokens for output in outputs if output.meta]
            )
            
            stage_tokens[subset] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        
        stage_info = {
            "created": last_output.created,
            "subsets": subsets,
            "tokens": stage_tokens,
            "batch_id": 
        }
    
    def _write_meta(self):
        model_info = {
            "model": self.llm_instance.model,
            "parameters": self.llm_instance.configs.model_dump()
        }
        
        stage_tokens
    
    
    def process(self):
        for stage in self.stages:
            stage_outputs: List[StageOutput] = list(
                filter(lambda output: output.stage_name == stage, self.output)
            )
            if stage in {"1", "1r", "1c"}:
                self._build_categories_pdf(stage, stage_outputs)
            else:
                self._build_data_output(stage, stage_outputs)