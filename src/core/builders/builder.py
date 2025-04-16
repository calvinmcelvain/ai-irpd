"""
Contains the Builder Model.
"""
import core.builders as builder
from _types.test_output import TestOutput



class Builder:
    """
    Builder model.
    
    Aggregates builders and builds output based on stage name.
    """
    def __init__(self, test_output: TestOutput):
        self.test_output = test_output

        
    def build(self, stage_name: str) -> None:
        if stage_name in {"1", "1r", "1c"}:
            builder.PDF(self.test_output).build(stage_name)
        else:
            builder.CSV(self.test_output).build(stage_name)
        return None
