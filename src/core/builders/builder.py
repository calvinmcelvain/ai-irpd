"""
Contains the Builder Model.
"""
import core.builders as builder
from helpers.utils import load_config
from _types.test_output import TestOutput


CONFIGS = load_config("irpd.json")["stage_classes"]



class Builder:
    """
    Builder model.
    
    Aggregates builders and builds output based on stage name.
    """
    def __init__(self, test_output: TestOutput):
        self.test_output = test_output

        
    def build(self, stage_name: str) -> None:
        if stage_name in CONFIGS["categorization"]:
            builder.PDF(self.test_output).build(stage_name)
        else:
            builder.CSV(self.test_output).build(stage_name)
        return None
