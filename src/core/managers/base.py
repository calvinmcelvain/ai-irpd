"""
Base model for manager models.
"""
from abc import ABC

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
        
