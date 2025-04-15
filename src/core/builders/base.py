"""
Contains the BaseBuilder model.
"""
from pathlib import Path
from abc import ABC, abstractmethod

from helpers.utils import load_config
from core.functions import output_attrb
from _types.test_output import TestOutput



class BaseBuilder(ABC):
    """
    BaseBuilder model.
    
    Sets the base attributes for builder models.
    """
    def __init__(self, test_output: TestOutput):
        self.sub_path = test_output.sub_path
        self.meta_path = test_output.meta_path
        self.meta = test_output.meta
        self.replication = test_output.replication
        self.llm_str = test_output.llm_str
        self.stage_outputs = test_output.stage_outputs
        
        self.irpd_config = self.meta.test_info
        self.cases = self.irpd_config.cases
        self.treatment = self.irpd_config.treatment
        self.ra = self.irpd_config.ra
        self.raw_data_path = Path(self.irpd_config.data_path) / "raw"
        
        self.output_attrb = output_attrb
        self.file_names = load_config("irpd.json")["output_file_names"]
    
    @abstractmethod
    def build(self, stage_name: str) -> None:
        """
        Builds a final output from outputs.
        """
        pass