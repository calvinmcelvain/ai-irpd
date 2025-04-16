"""
Contains the Data model.
"""
import pandas as pd

from core.foundation import FoundationalModel
from _types.irpd_config import IRPDConfig
from _types.test_output import TestOutput



class Data(FoundationalModel):
    """
    Data model.
    
    Used to filter RA data, build data, & bootstrap summary data.
    """
    def __init__(self, irpd_config: IRPDConfig):
        super().__init__(irpd_config)
        
        self.ra_data = self._pre_filter()
    
    def _pre_filter(self) -> pd.DataFrame:
        """
        Pre-filters RA data for the IRPDConfig.
        """
        # importing RA data
        df = pd.read_csv(self.data_path / "ra_summaries.csv")
        
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
        
    def filter_ra_data(self, subset: str) -> pd.DataFrame:
        """
        Filter's the RA data for a given subset & drops unneeded columns.
        """
        df = self.ra_data
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
        df = self.ra_data
        
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
        
        
        
        
        