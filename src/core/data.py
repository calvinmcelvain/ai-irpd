"""
Contains the Data model.
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from helpers.utils import load_config
from core.foundation import FoundationalModel
from _types.irpd_config import IRPDConfig
from _types.test_output import TestOutput


CONFIGS = load_config("irpd.json")



class Data(FoundationalModel):
    """
    Data model.
    
    Used to filter RA data, build data, & bootstrap summary data.
    """
    def __init__(self, irpd_config: IRPDConfig):
        super().__init__(irpd_config)
        
        self.ra_data = self._pre_filter_ra()
        self.raw_data = self._pre_filter_raw()
        
    def _get_summary_data(self, sub_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Returns the df for summary data, depending on the ra type.
        """
        if self.ra == "llm":
            assert isinstance(sub_path, Path), (
                f"`sub_path` must be a Path object. Got {sub_path}")
            df = sub_path / CONFIGS["output_file_names"]["summaries"]["0"]
        else:
            df = self.ra_data
        return df
    
    def _pre_filter_ra(self) -> pd.DataFrame:
        """
        Pre-filters RA data for the IRPDConfig.
        """
        # importing RA data
        df = pd.read_csv(self.data_path / CONFIGS["file_paths"]["ra"])
        
        # Filter for cases and treatment.
        df = df[
            (df["case"].isin(self.cases)) &
            ((self.treatment == "merged") | (df["treatment"] == self.treatment))
        ]
        
        # Drop unnecessary RA columns if a singular RA is specified.
        if self.ra != "both":
            ra_cols_to_drop = [
                f"summary_{ra}" for ra in ["ra1", "ra2"] if ra != self.ra
            ]
            df = df.drop(columns=ra_cols_to_drop, errors="ignore")
        
        return df
    
    def _pre_filter_raw(self) -> pd.DataFrame:
        """
        Pre-filters Raw data for IRPDConfig.
        """
        # importing raw data.
        df = pd.read_csv(self.data_path / CONFIGS["file_paths"]["llm"], index_col=0)
        
        # Filter for treatment.
        df = df[(
            (self.treatment == "merged") | (df["treatment"] == self.treatment)
        )]
        
        # Adjusting for max summaries.
        if self.max_summaries: df = df[:self.max_summaries]
        
        return df
    
    def get_list_of_raw_instances(self) -> List[List[Dict]]:
        """
        Returns a list of instances, given the context.
        """
        df = self.raw_data
        
        # Getting all windows.
        window_numbers = df.loc[
            df["case"].isin(self.cases), "window_number"].dropna().tolist()
        
        # Getting keep columns based on treatment
        keep_columns = [
            "window_number", "chat", "super_game", "stage_game", "payoff",
            "cooperate", "other_cooperate", "team", "opponent", "sender",
            "receiver", "instance_type"
        ]
        if self.treatment != "perfect":
            keep_columns.extend([
                "cooperate_signal", "other_cooperate_signal"
            ])
        
        raw_instances = []
        for window_number in window_numbers:
            start_idx = df[df["window_number"] == window_number].index[0]
            ref_team = df.loc[start_idx, 'team']
            
            # Get the starting point's super_game and stage_game.
            stage_forward = 0
            end_idx = start_idx
            prev_super = df.loc[start_idx, 'super_game']
            prev_stage = df.loc[start_idx, 'stage_game']

            # Getting context forward.
            while stage_forward < self.context[1] and end_idx < len(df) - 1:
                end_idx += 1
                row = df.loc[end_idx]
                if row['team'] != ref_team:
                    end_idx -= 1
                    break

                curr_super = row['super_game']
                curr_stage = row['stage_game']

                if (curr_super != prev_super) or (curr_stage != prev_stage):
                    stage_forward += 1
                    prev_super = curr_super
                    prev_stage = curr_stage

            stage_backward = 0
            back_idx = start_idx
            prev_super = df.loc[start_idx, 'super_game']
            prev_stage = df.loc[start_idx, 'stage_game']

            # Getting context backward.
            while stage_backward < self.context[0] and back_idx > 0:
                back_idx -= 1
                row = df.loc[back_idx]
                if row['team'] != ref_team:
                    back_idx += 1
                    break

                curr_super = row['super_game']
                curr_stage = row['stage_game']

                if (curr_super != prev_super) or (curr_stage != prev_stage):
                    stage_backward += 1
                    prev_super = curr_super
                    prev_stage = curr_stage
            
            subset_df = df.loc[back_idx:end_idx]
            subset_df = subset_df[keep_columns]
            
            raw_instances.append(subset_df.to_dict("records"))
        return raw_instances
        
        
    def filter_summary_data(self, subset: str, sub_path: Path = None) -> pd.DataFrame:
        """
        Filter's the summary data for a given subset & drops unneeded columns.
        """
        df = self._get_summary_data(sub_path)
        if subset != "full": df = df[(df["subset"] == subset)]
        df = df.drop(columns=["case", "treatment", "subset"])
        return df
    
    def adjust_for_completed_outputs(
        self,
        test_output: TestOutput,
        stage_name: str
    ) -> pd.DataFrame:
        """
        Adjusts data for requests/outputs that have already been made. Used
        for iterative stages (stage 0, 2, & 3).
        """
        df = self._get_summary_data(test_output.sub_path)
        
        # Adjusting for max_instances
        if self.max_instances: df = df[:self.max_instances]
        
        # Adjusting for completed request (if any).
        outputs = test_output.stage_outputs[stage_name].outputs["full"]
        if outputs:
            window_nums = [
                output.request_out.parsed.window_number
                for output in outputs
            ]
            df = df[~df["window_number"].isin(window_nums)]
        
        return df
    
    
        
        
        
        