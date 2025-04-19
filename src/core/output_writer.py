"""
Contains the OutputWriter model.
"""
import logging

from helpers.utils import write_json
from core.builders.builder import Builder
from core.foundation import FoundationalModel
from _types.irpd_config import IRPDConfig
from _types.test_output import TestOutput


log = logging.getLogger("app")



class OutputWriter(FoundationalModel):
    """
    OutputWriter model.
    
    Writes the raw prompts & responses, as well as the test meta and 
    final output forms.
    """
    def __init__(self, irpd_config: IRPDConfig):
        super().__init__(irpd_config)
    
    def write_output(
        self, test_output: TestOutput, stage_name: str, subset: str
    ) -> None:
        """
        Writes the raw prompts, responses, & meta. Also writes the final form 
        outputs if stage complete.
        """
        self.write_meta(test_output)
        
        stage_output = test_output.stage_outputs[stage_name]

        for output in stage_output.outputs[subset]: output.write()
        
        if stage_output.complete: Builder(test_output).build(stage_name)
        
        return None
        
        
    def write_meta(self, test_output: TestOutput) -> None:
        """
        Writes the meta for given TestOutput object.
        """
        meta_path = test_output.meta_path
        meta = test_output.meta.model_dump()
        
        write_json(meta_path, meta)
        
        return None
