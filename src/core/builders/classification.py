"""
Contains the ClassificationCSV model
"""
import logging
import pandas as pd
from typing import List

from helpers.utils import load_config
from core.builders.base_builder import BaseBuilder


CONFIGS = load_config("irpd.json")


log = logging.getLogger("app")



class ClassificationCSV(BaseBuilder):
    """
    ClassificationCSV model.
    
    Builds classification CSVs from outputs.
    """
    def build(self, stage_name: str) -> None:
        """
        Builds the final form classification CSV files for Stage 2 & 3.
        
        Saves to subpath directory.
        """
        # Get summary data files for each case.
        if self.ra == "llm":
            og_df = self._load_original_raw_data()
        else:
            og_dfs = self._load_original_ra_data()

        # Build df from outputs.
        output_df = self._process_stage_outputs(stage_name)

        # Merge responses with original summary files.
        if self.ra == "llm":
            final_df = pd.merge(og_df, output_df, on="window_number", how="left")
        else:
            final_df = self._merge_with_original_ra_data(og_dfs, output_df)
        
        # Save the final DataFrame to a CSV file
        csv_path = self.sub_path / self.file_names["classifications"][stage_name]
        final_df.to_csv(csv_path, index=False)

    def _load_original_ra_data(self) -> List[pd.DataFrame]:
        """
        Loads original summary RA data files for each case.
        """
        og_df_names = [
            f"{case}_{self.treatment}_{self.ra}.csv"
            for case in self.cases
        ]
        if self.max_instances: 
            return [
                pd.read_csv(self.data_path / "raw" / name)[:self.max_instances]
                for name in og_df_names
            ]
        return [pd.read_csv(self.raw_data_path / name) for name in og_df_names]
    
    def _load_original_raw_data(self) -> List[pd.DataFrame]:
        """
        Loads original raw data file for each case.
        """
        og_df_name = self.sub_path / CONFIGS["file_paths"]["llm"]
        df = pd.read_csv(self.data_path / og_df_name)
        df = df["window_number"].dropna()
        if self.max_instances: 
            return df[:self.max_instances]
        return df

    def _process_stage_outputs(self, stage_name: str) -> pd.DataFrame:
        """
        Processes stage outputs and creates a DataFrame.
        """
        outputs = self.stage_outputs[stage_name].outputs["full"]
        output_list = []

        for output in outputs:
            output_parsed = output.parsed
            response = {
                "window_number": output_parsed.window_number,
                "reasoning": output_parsed.reasoning
            }
            
            # Binary classification for stage 2 and rank for stage 3.
            for category in self.output_attrb(output_parsed):
                response[category.category_name] = getattr(category, "rank", 1)

            output_list.append(response)

        return pd.DataFrame.from_records(output_list).fillna(0)

    def _merge_with_original_ra_data(
        self, og_dfs: List[pd.DataFrame], output_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges processed outputs with original RA summary data.
        """
        final_dfs = []

        for case, og_df in zip(self.cases, og_dfs):
            merged_df = pd.merge(og_df, output_df, on="window_number", how="left")
            merged_df["case"] = case
            final_dfs.append(merged_df)

        return pd.concat(final_dfs, ignore_index=True, sort=False)

