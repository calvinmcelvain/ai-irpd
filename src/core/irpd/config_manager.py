"""
Test config manager.

Contains the function model, ConfigManager.
"""
import logging

from helpers.utils import create_directory
from core.irpd.functions import instance_types
from core.llms.models import LLMModel
from types.test_config import TestConfig


log = logging.getLogger(__name__)



class ConfigManager:
    """
    ConfigManger model.
    
    Used to generate subpaths, meta-paths, subsets, and llm instances for a 
    given TestConfig.
    """
    def __init__(self, test_config: TestConfig):
        self.config = test_config
        self.stages = test_config.stages
        self.cases = test_config.cases
        self.test_path = test_config.test_path
        self.llms = test_config.llms
        self.llm_config = test_config.llm_config
        self.total_replications = test_config.total_replications
    
    def generate_subpath(self, n: int, llm_str: str):
        """
        Generates a subpath for a given replication and LLM. 
        
        For tests w/ more than one LLM, a dir. is made for each LLM. For tests 
        w/ more than one replication, a dir. is made for each replicaiton (w/ 
        respect to the LLM dir.)
        """
        subpath = self.test_path
        if len(self.llms) > 1: subpath = subpath / llm_str
        if self.total_replications > 1: subpath = subpath / f"replication_{n}"
        if not subpath.exists():
            create_directory(subpath)
        return subpath
    
    def generate_meta_path(self, n: int, llm_str: str):
        """
        Generates the path for 'test' meta. File exists for each subpath.
        """
        subpath = self.generate_subpath(n, llm_str)
        return subpath / "_test_meta.json"
    
    def get_subsets(self, stage_name: str):
        """
        Generates subsets for a given stage.
        """
        subsets = ["full"]
        if stage_name in {"1", "1r"}:
            prod = [
                (case, instance_type)
                for case in self.cases
                for instance_type in instance_types(case)
            ]
            subsets += [f"{c}_{i}" for c, i in prod]
        return subsets
    
    def generate_llm_instance(self, llm_str: str, print_reponse: bool = False):
        """
        Returns the LLM model instance from the /llms package.
        """
        return getattr(LLMModel, llm_str).get_llm_instance(self.llm_config, print_reponse)