"""
Base model for manager models.
"""
from abc import ABC

from helpers.utils import create_directory
from core.functions import instance_types
from core.llms.llm_models import LLMModel
from types.irpd_config import IRPDConfig



class Manager(ABC):
    """
    Abstract Manager class for manager inheritance.
    
    Fuck composition (jkjkjkjkjk).
    """
    def __init__(
        self,
        irpd_config: IRPDConfig
    ):
        self.irpd_config = irpd_config
        self.stages = irpd_config.stages
        self.cases = irpd_config.cases
        self.test_path = irpd_config.test_path
        self.llms = irpd_config.llms
        self.llm_config = irpd_config.llm_config
        self.total_replications = irpd_config.total_replications
        
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