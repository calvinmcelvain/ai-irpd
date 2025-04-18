"""
Contains the SummaryCSV model.
"""
import logging
import pandas as pd
from typing import List

from helpers.utils import load_config
from core.builders.base_builder import BaseBuilder


CONFIGS = load_config("irpd.json")


log = logging.getLogger("app")




class SummaryCSV(BaseBuilder):
    """
    SummaryCSV model.
    
    Builds summary CSVs from outputs in Stage 0.
    """
    def build(self, stage_name: str = "0"):
        """
        Builds the summary CSV for stage 0.
        """
        summary_df = self._process_stage_outputs(stage_name)
        final_df = self._merge_with_original_data(summary_df)
        
        # Save the final DataFrame to a CSV file
        csv_path = self.sub_path / self.file_names["summaries"][stage_name]
        final_df.to_csv(csv_path, index=False)
        log.info(f"Stage {stage_name} CSV saved to: {csv_path}")
    
    def _process_stage_outputs(self, stage_name: str) -> pd.DataFrame:
        """
        Processes stage 0 outputs and creates a DataFrame.
        """
        outputs = self.stage_outputs[stage_name].outputs
        output_list = []

        for _, output in outputs.values():
            output_parsed = output.parsed
            response = {
                "window_number": output_parsed.window_number,
                "summary": output_parsed.summary
            }
            output_list.append(response)

        return pd.DataFrame.from_records(output_list)
    
    def _merge_with_original_data(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Loads original raw data and merges with summary df.
        """
        og_df = pd.read_csv(self.data_path / CONFIGS["file_paths"]["llm"])
        og_df = og_df["window_number"].dropna()
        og_df = og_df[og_df["case"].isin()]
        
        merged_df = pd.merge(og_df, summary_df, on="window_number", how="left")
        
        keep_columns = [
            "window_number", "summary", "instance_type",
            "case", "subset", "treatment"
        ]
        
        merged_df = merged_df[keep_columns]
        merged_df = merged_df["summary"].dropna()
        
        return merged_df
        
        