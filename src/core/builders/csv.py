"""
Contains the CSV model
"""
import logging
import pandas as pd
from typing import List

from core.builders.base import BaseBuilder


log = logging.getLogger(__name__)



class CSV(BaseBuilder):
    """
    CSV model.
    
    Builds category CSVs from outputs.
    """
    def build(self, stage_name: str) -> None:
        """
        Builds the final form classification CSV files for Stage 0, 2 & 3.
        
        Saves to subpath directory.
        """
        if "0" in self.stage_outputs:
            log.error("Stage 0 not setup yet.")
            return

        # Get summary data files for each case.
        og_dfs = self._load_original_data()

        # Build df from outputs.
        output_df = self._process_stage_outputs(stage_name)

        # Merge responses with original summary files.
        final_df = self._merge_with_original_data(og_dfs, output_df)

        # Save the final DataFrame to a CSV file
        csv_path = self.sub_path / self.file_names["classifications"][stage_name]
        final_df.to_csv(csv_path, index=False)

    def _load_original_data(self) -> List[pd.DataFrame]:
        """
        Loads original summary data files for each case.
        """
        og_df_names = [
            f"{case}_{self.treatment}_{self.ra}.csv" for case in self.cases
        ]
        return [pd.read_csv(self.raw_data_path / name) for name in og_df_names]

    def _process_stage_outputs(self, stage_name: str) -> pd.DataFrame:
        """
        Processes stage outputs and creates a DataFrame.
        """
        outputs = self.stage_outputs[stage_name].outputs["full"]
        output_list = []

        for output in outputs:
            output_parsed = output.request_out.parsed
            response = {
                "reasoning": output_parsed.reasoning,
                "window_number": output_parsed.window_number,
            }

            # Binary classification for stage 2 and rank for stage 3.
            for category in self.output_attrb(output_parsed):
                response[category.category_name] = getattr(category, "rank", 1)

            output_list.append(response)

        return pd.DataFrame.from_records(output_list).fillna(0)

    def _merge_with_original_data(
        self, og_dfs: List[pd.DataFrame], output_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges processed outputs with original summary data.
        """
        final_dfs = []

        for case, og_df in zip(self.cases, og_dfs):
            merged_df = pd.merge(og_df, output_df, on="window_number", how="left")
            merged_df["case"] = case
            final_dfs.append(merged_df)

        return pd.concat(final_dfs, ignore_index=True, sort=False)

