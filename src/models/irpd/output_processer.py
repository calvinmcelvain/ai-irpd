"""
Output processing module.

Contains the OutputProcessor model.
"""
import logging
import json
import pandas as pd
from pathlib import Path
from typing import List
from datetime import datetime

from utils import txt_to_pdf, load_json_n_validate, write_json, write_file, create_directory
from tools.functions import categories_to_txt, output_attrb
from models.irpd.test_outputs import StageOutput, ModelInfo, StageInfo, SubsetInfo, TestMeta
from models.irpd.config_manager import ConfigManager


log = logging.getLogger(__name__)



class OutputProcesser:
    """
    OutputProcessor model.
    
    Takes a list of StageOutput objects and writes outputs and meta data.
    
    Also handles the final form outputs:
        - Stage 0 summary CSV (not complete).
        - Stage 1, 1r, & 1c PDFs.
        - Stage 2 & 3 CSVs.
    """
    def __init__(
        self,
        stage_outputs: List[StageOutput],
        config_manager: ConfigManager
    ):
        self.outputs = stage_outputs
        self.config_manager = config_manager
        self.test_config = config_manager.config
        self.stage_name = stage_outputs[0].stage_name
        self.stages = config_manager.stages
        self.total_replications = config_manager.total_replications
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
    
    def _build_categories_pdf(self):
        """
        Builds the final form Category PDF files for Stage 1, 1r, & 1c.
        
        Saves PDF to subpath.
        """
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
    
    def _build_classification_output(self):
        """
        Builds the final form calssification CSV files for Stage 2 & 3.
        
        Saves to subpath directory.
        """
        dfs = []
        for case in self.cases:
            
            # Getting the original summary data file.
            if "0" not in self.stages:
                raw_path = self.data_path / "raw"
                og_df_path = raw_path / f"{case}_{self.treatment}_{self.ra}.csv"
                og_df = pd.read_csv(og_df_path)
            else:
                log.error("Stage 0 not setup yet.")
            
            response_list = []
            
            # Because Stage 2 & 3 are done via one subset, should only have one 
            # output.
            for output in self.outputs[0].outputs:
                response = {"reasoning": output.parsed.reasoning}
                response["window_number"] = output.parsed.window_number
                for cat in output_attrb(output.parsed):
                    response[cat.category_name] = 1
                    # If Stage 2, replace binary to rank.
                    if hasattr(cat, "rank"):
                        response[cat.category_name] = cat.rank
                # Appending each response as a record.
                response_list.append(response)

            df = pd.DataFrame.from_records(response_list).fillna(0)
            
            # Merging responses w/ og summary file.
            merged_df = pd.merge(og_df, df, on='window_number')
            merged_df["case"] = case
            
            # Appending the 'case' specific df.
            dfs.append(merged_df)
        
        # Concatenating case dfs, for tests w/ a composition of cases (e.g., 
        # uni_switch case).
        df = pd.concat(dfs, ignore_index=True, sort=False)
        df.to_csv(self.sub_path / f"_stage_{self.stage_name}_final.csv", index=False)
        return None
    
    def _write_output(self):
        """
        Writes the raw output & prompt files.
        """
        for output in self.outputs:
            # If output complete, no need to rewrite the output.
            if not output.complete and output.outputs:
                prompts_path = self.stage_path / output.subset / "prompts"
                responses_path = self.stage_path / output.subset / "responses"
                create_directory([prompts_path, responses_path])
                
                # For stages 0, 2 & 3, the system prompt is static. Highly 
                # redundant to write the system prompt for len(responses).
                system_prompt = output.outputs[0].prompts.system
                system_path = f"{output.subset}_stg_{self.stage_name}_system_prompt.txt"
                write_file(prompts_path / system_path, system_prompt)
                
                for response in output.outputs:
                    user_prompt = response.prompts.user
                    
                    # Stage 2 & 3 reponse files are differentiated via window
                    # number.
                    user_path = f"{output.subset}_"
                    response_path = f"{output.subset}_"
                    if self.stage_name in {"2", "3"}:
                        user_path += f"{response.parsed.window_number}_user_prompt.txt"
                        response_path += f"{response.parsed.window_number}_response.txt"
                    else:
                        user_path += f"stg_{self.stage_name}_user_prompt.txt"
                        response_path += f"stg_{self.stage_name}_response.txt"
                    
                    # Responses written in a more readable JSON structure.
                    write_file(prompts_path / user_path, user_prompt)
                    write_file(
                        responses_path / response_path,
                        json.dumps(response.parsed.model_dump(), indent=2)
                    )
        return None
    
    def _stage_meta_info(self, meta: TestMeta):
        """
        Creates the stage specific meta info.
        """
        # Initializing StageInfo object if not.
        if self.stage_name not in meta.stages.keys():
            meta.stages[self.stage_name] = StageInfo()
        
        stage_info = meta.stages[self.stage_name]
        stage_info.batch_id = self.batch_id
        
        # Cannot write Path object.
        stage_info.batch_path = self.batch_path
        if self.batch_path: stage_info.batch_path = self.batch_path.as_posix()
        
        for output in self.outputs:
            # If stage is complete, it should mean that the meta has already
            # been written.
            if not output.complete and output.outputs:
                # Creating SubsetInfo object.
                # Note: `created` attrb. is written as a unix timestamp.
                stage_info.subsets[output.subset] = SubsetInfo(
                    created=str(datetime.fromtimestamp(output.outputs[0].meta.created)),
                    input_tokens=sum([t.meta.input_tokens for t in output.outputs]),
                    output_tokens=sum([t.meta.output_tokens for t in output.outputs]),
                    total_tokens=sum([t.meta.total_tokens for t in output.outputs])
                )
        return stage_info
    
    def write_meta(self):
        """
        Writes the TestMeta object for a given StageOutput obj.
        
        Written as a JSON in the subpath dir.
        """
        # Load meta if exists. Else, create TestMeta obj.
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
        
        # Write stage specific meta info.
        meta.stages[self.stage_name] = self._stage_meta_info(meta)
        
        write_json(self.meta_path, meta.model_dump())
        log.info(
            f"\nMeta data successully written for:"
            f"\n\t config: {self.test_config.id}"
            f"\n\t case: {self.test_config.case}"
            f"\n\t llm: {self.llm_str}"
            f"\n\t replicate: {self.replication} of {self.total_replications}"
            f"\n\t stage: {self.stage_name}"
        )
        return None
        
    def process(self, stage_complete: bool = False):
        """
        Writes outputs & meta. If stage_complete True, writes the final forms
        for the given stage (defined on initialization).
        """
        self._write_output()
        self.write_meta()
        if stage_complete:
            if self.stage_name in {"1", "1r", "1c"}:
                self._build_categories_pdf()
            else:
                self._build_classification_output()
        return None