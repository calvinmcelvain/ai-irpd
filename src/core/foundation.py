"""
Contains the FoundationalModel
"""
from abc import ABC
from typing import List
from pathlib import Path

from helpers.utils import create_directory, dynamic_import, load_config
from core.functions import instance_types
from core.llms.clients.base import BaseLLM
from core.llms.llm_models import LLMModel
from _types.irpd_config import IRPDConfig



class FoundationalModel(ABC):
    def __init__(self, irpd_config: IRPDConfig):
        self.print_response = False # TestRunner sets this again.
        self.irpd_config = irpd_config
        self.stages = irpd_config.stages
        self.cases = irpd_config.cases
        self.batches = irpd_config.batches
        self.test_path = Path(irpd_config.test_path)
        self.data_path = Path(irpd_config.data_path)
        self.prompts_path = Path(irpd_config.prompts_path)
        self.max_instances = irpd_config.max_instances
        self.llms = irpd_config.llms
        self.case = irpd_config.case
        self.ra = irpd_config.ra
        self.treatment = irpd_config.treatment
        self.llm_config = irpd_config.llm_config
        self.total_replications = irpd_config.total_replications
        
        self.schemas = {
            stage: dynamic_import("_types.stage_schemas", f"Stage{stage}Schema")
            for stage in self.stages
        }
        self.subsets = {
            stage: self._get_subsets(stage)
            for stage in self.stages
        }
        self.llm_instances = {
            llm_str: self._generate_llm_instance(llm_str)
            for llm_str in self.llms
        }
        self.file_names = load_config("irpd.json")["output_file_names"]
        
    def _generate_subpath(self, n: int, llm_str: str) -> Path:
        """
        Generates a subpath for a given replication and LLM. 
        
        For tests w/ more than one LLM, a dir. is made for each LLM. For tests 
        w/ more than one replication, a dir. is made for each replicaiton (w/ 
        respect to the LLM dir.)
        """
        subpath = self.test_path
        if len(self.llms) > 1: subpath = subpath / llm_str
        if self.total_replications > 1: subpath = subpath / f"replication_{n}"
        if not subpath.exists(): create_directory(subpath)
        return subpath
    
    def _get_subsets(self, stage_name: str) -> List[str]:
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
    
    def _generate_llm_instance(self, llm_str: str) -> BaseLLM:
            """
            Returns the LLM model instance from the /llms package.
            """
            return getattr(LLMModel, llm_str).get_llm_instance(
                self.llm_config, self.print_reponse)