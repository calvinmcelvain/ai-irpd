import logging

from models.irpd.test_config import TestConfig
from models.llm_model import LLMModel


log = logging.getLogger(__name__)



class TestRunner:
    def __init__(
        self,
        config: TestConfig,
        subpath_function: function,
        print_response: bool = False
    ):
        self.stages = config.stages
        self.batch_request = config.batches
        self.llm_config = config.llm_config
        self.replications = config.total_replications
        self.llms = config.llms
        self.print_response = print_response
        self.get_sub_path = subpath_function
        
    
    def _generate_llm_instance(self, llm: str):
        return getattr(LLMModel, llm).get_llm_instance(
            self.llm_config, self.print_response
        )
    
    def _run_batch(self):
        pass
    
    def _run_completion(self):
        pass
    
    def run(self):
        pass