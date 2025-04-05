import logging
import pandas as pd
from typing import List

from utils import txt_to_pdf, load_json_n_validate, write_json, write_file
from tools.functions import categories_to_txt, output_attrb
from models.irpd.outputs import StageOutput, ModelInfo, StageInfo, StageTokens, TestMeta


log = logging.getLogger(__name__)



class OutputProcesser:
    def __init__(self, stage_outputs: List[StageOutput]):
        self.outputs = stage_outputs
        self.configs = stage_outputs[0].stage_config
        self.stage = self.configs.stage_name
        self.sub_path = self.configs.sub_path
        self.meta_path = self.configs.meta_path
        self.cases = self.configs.cases
        self.treatment = self.configs.treatment
        self.ra = self.configs.ra
        self.data_path = self.configs.data_path
        self.llm_instance = self.configs.llm_instance
        self.batch_id = self.configs.batch_id
        self.subsets = list(set(output.subset for output in stage_outputs))
    
    def _build_categories_pdf(self):
        pdf = f"# Stage {self.stage_name} Categories\n\n"
        for output in self.outputs:
            categories = output_attrb(output.outputs[0].parsed)
            
            subset = output.stage_config.subset
            
            if subset != "full":
                case, instance_type = subset.split("_")
                pdf += f"## {case.capitalize()} - {instance_type.upper()} Categories\n\n"
            else:
                if self.stage_name == "1c":
                    pdf += f"## Final Category Set\n\n"
                else:
                    pdf += f"## Unified Categories\n\n"
            pdf += categories_to_txt(categories)
        pdf_path = self.sub_path / f"_stage_{self.stage_name}_categories.pdf"
        txt_to_pdf(pdf, pdf_path)
        return None
    
    def _build_data_output(self):
        dfs = []
        for case in self.cases:
            raw_path = self.data_path / "raw"
            raw_df_path = raw_path / f"{case}_{self.treatment}_{self.ra}.csv"
            raw_df = pd.read_csv(raw_df_path)
            df_list = []
            
            for outputs in self.outputs:
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
        df.to_csv(self.sub_path / f"_stage_{self.stage_name}_final.csv", index=False)
        return None
    
    def _stage_info(self, meta: TestMeta):
        stage_info = meta.stages[self.stage_name]
        stage_info.created = self.outputs[0].outputs[0].meta.created
        stage_info.subsets.extend(self.subsets)
        stage_info.batch_id = self.batch_id
        
        for output in self.outputs:
            input_tokens = sum([t.meta.input_tokens for t in output.outputs])
            output_tokens = sum([t.meta.output_tokens for t in output.outputs])
            total_tokens = sum([input_tokens, output_tokens])
            
            stage_info_tokens = stage_info.tokens[output.subset]
            stage_info_tokens.input_tokens += input_tokens
            stage_info_tokens.output_tokens += output_tokens
            stage_info_tokens.total_tokens += total_tokens
        return meta
            
    def _write_meta(self):
        if self.meta_path.exists():
            meta = load_json_n_validate(self.meta_path, TestMeta)
        else:
            model_info = ModelInfo(
                model=self.llm_instance.model,
                parameters=self.llm_instance.configs.model_dump_json()
            )
            tokens = {subset: StageTokens() for subset in self.subsets}
            meta = TestMeta(
                model_info=model_info,
                sub_config=self.configs,
                stages={self.stage_name: StageInfo(tokens=tokens)}
            )
        
        meta = self._stage_info(meta)
        write_json(self.meta_path, meta)
        return None
    
    def _write_output(self):
        for output in self.outputs:
            if not output.complete:
                prompts_path = self.configs.prompts_path
                responses_path = self.configs.responses_path
                system_prompt = output.outputs[0].prompts.system
                system_path = f"{self.subsets[0]}_stg_{self.stage}_system_prompt.txt"
                write_file(prompts_path / system_path, system_prompt)
                
                for response in output.outputs:
                    user_prompt = response.prompts.user
                    user_path = f"{self.subsets[0]}_"
                    response_path = f"{self.subsets[0]}_"
                    if self.stage in {"2", "3"}:
                        user_path += f"{response.parsed.window_number}_user_prompt.txt"
                        response_path += f"{response.parsed.window_number}_response.txt"
                    else:
                        user_path += f"stg_{self.stage}_user_prompt.txt"
                        response_path += f"stg_{self.stage}_response.txt"
                    write_file(prompts_path / user_path, user_prompt)
                    write_file(responses_path / response_path, response.text)
            return None
    
    def process(self, stage_complete: bool):
        self._write_output()
        self._write_meta()
        if stage_complete:
            if self.stage_name in {"1", "1r", "1c"}:
                self._build_categories_pdf()
            else:
                self._build_data_output()
        return None