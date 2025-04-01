import logging
from pathlib import Path

from models.irpd.test_config import TestConfig
from models.llm_model import LLMModel


log = logging.getLogger(__name__)



class TestRunner:
    def __init__(
        self,
        config: TestConfig,
        llm: str,
        sub_path: Path,
        print_response: bool = False
    ):
        self.stages = config.stages
        self.batch_request = config.batches
        self.llm_config = config.llm_config
        
        self.llm = self._generate_llm_instance(llm)
        self.print_response = print_response
        self.sub_path = sub_path
        
    
    def _generate_llm_instance(self, llm: str):
        return getattr(LLMModel, llm).get_llm_instance(
            self.llm_config, self.print_response
        )
    
    def run(self):
        pass