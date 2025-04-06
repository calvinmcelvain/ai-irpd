import logging
import json
import pandas as pd
from typing import List
from datetime import datetime

from utils import txt_to_pdf, load_json_n_validate, write_json, write_file, create_directory
from tools.functions import categories_to_txt, output_attrb
from models.irpd.test_outputs import StageOutput, ModelInfo, StageInfo, SubsetInfo, TestMeta
from models.irpd.config_manager import ConfigManager


log = logging.getLogger(__name__)



class OutputProcesser:
    def __init__(
        self,
        stage_outputs: List[StageOutput],
        config_manager: ConfigManager
    ):
        self.outputs = stage_outputs
        self.config_manager = config_manager
        self.stage_name = stage_outputs[0].stage_name
        self.replication = stage_outputs[0].replication
        self.llm_str = stage_outputs[0].llm_str
        self.sub_path = config_manager.generate_subpath(self.replication, self.llm_str)
        self.meta_path = config_manager.generate_meta_path(self.replication, self.llm_str)
        self.stage_path = self.sub_path / f"stage_{self.stage_name}"
        self.cases = config_manager.config.cases
        self.treatment = config_manager.config.treatment
        self.ra = config_manager.config.ra
        self.data_path = config_manager.config.data_path
        self.llm_instance = config_manager.generate_llm_instance(self.llm_str)
        self.batch_id = stage_outputs[0].batch_id
        self.batch_path = stage_outputs[0].batch_path
        self.subsets = list(set(output.subset for output in stage_outputs))
    
    def _build_categories_pdf(self):
        pdf = f"# Stage {self.stage_name} Categories\n\n"
        for output in self.outputs:
            categories = output_attrb(output.outputs[0].parsed)
            
            subset = output.subset
            
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
                    for cat in output_attrb(output.parsed):
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
    
    def _write_output(self):
        for output in self.outputs:
            if not output.complete and output.outputs:
                prompts_path = self.stage_path / output.subset / "prompts"
                responses_path = self.stage_path / output.subset / "responses"
                create_directory([prompts_path, responses_path])
                
                system_prompt = output.outputs[0].prompts.system
                system_path = f"{output.subset}_stg_{self.stage_name}_system_prompt.txt"
                write_file(prompts_path / system_path, system_prompt)
                
                for response in output.outputs:
                    user_prompt = response.prompts.user
                    user_path = f"{output.subset}_"
                    response_path = f"{output.subset}_"
                    if self.stage_name in {"2", "3"}:
                        user_path += f"{response.parsed.window_number}_user_prompt.txt"
                        response_path += f"{response.parsed.window_number}_response.txt"
                    else:
                        user_path += f"stg_{self.stage_name}_user_prompt.txt"
                        response_path += f"stg_{self.stage_name}_response.txt"
                    write_file(prompts_path / user_path, user_prompt)
                    write_file(
                        responses_path / response_path,
                        json.dumps(response.parsed.model_dump(), indent=2)
                    )
        return None
    
    def _stage_info(self, meta: TestMeta):
        if self.stage_name not in meta.stages.keys():
            meta.stages[self.stage_name] = StageInfo()
        stage_info = meta.stages[self.stage_name]
        stage_info.batch_id = self.batch_id
        stage_info.batch_path = self.batch_path.as_posix()
        
        for output in self.outputs:
            if not output.complete and output.outputs:
                stage_info.subsets[output.subset] = SubsetInfo()
                time = datetime.fromtimestamp(output.outputs[0].meta.created)
                stage_info.subsets[output.subset].created = str(time)
                
                input_tokens = sum([t.meta.input_tokens for t in output.outputs])
                output_tokens = sum([t.meta.output_tokens for t in output.outputs])
                total_tokens = sum([input_tokens, output_tokens])
                
                stage_info.subsets[output.subset].input_tokens += input_tokens
                stage_info.subsets[output.subset].output_tokens += output_tokens
                stage_info.subsets[output.subset].total_tokens += total_tokens
        return stage_info
    
    def write_meta(self):
        if self.meta_path.exists():
            meta = load_json_n_validate(self.meta_path, TestMeta)
        else:
            model_info = ModelInfo(
                model=self.llm_instance.model,
                parameters=self.llm_instance.configs.model_dump()
            )
            stages = {stage: StageInfo() for stage in self.config_manager.stages}
            meta = TestMeta(
                model_info=model_info,
                test_info=self.config_manager.config.convert_to_dict(),
                stages=stages
            )
        
        meta.stages[self.stage_name] = self._stage_info(meta)
        write_json(self.meta_path, meta.model_dump())
        return None
        
    def process(self, stage_complete: bool = False):
        self._write_output()
        self.write_meta()
        if stage_complete:
            if self.stage_name in {"1", "1r", "1c"}:
                self._build_categories_pdf()
            else:
                self._build_data_output()
        return None