import logging
from pathlib import Path

from models.irpd.test_config import TestConfig
from models.llm_model import LLMModel


log = logging.getLogger(__name__)



class TestRunner:
    def __init__(
        self,
        config: TestConfig,
        print_response: bool = False
    ):
        self.stages = config.stages
        self.batch_request = config.batches
        self.llm_config = config.llm_config
        self.test_path = config.test_path
        self.total_replications = config.total_replications
        self.replications = config.total_replications
        self.llms = config.llms
        self.print_response = print_response
        
    def _generate_subpath(self, N: int, llm_str: str):
        subpath = self.test_path
        if len(self.llms) > 1: subpath = subpath / llm_str
        if self.total_replications > 1: subpath = subpath / f"replication_{N}"
        return subpath
        
    
    def _generate_llm_instance(self, llm: str):
        return getattr(LLMModel, llm).get_llm_instance(
            self.llm_config, self.print_response
        )
    
    def _run_batch(self):
        pass
    
    def _run_completions(self):
        pass
    
    def run(self):
        pass