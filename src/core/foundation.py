"""
Module contains the FoundationModel.
"""
from abc import ABC
from pathlib import Path

from helpers.utils import lazy_import, create_directory, load_config
from core.functions import instance_types
from types.irpd_config import IRPDConfig


FILE_NAMES = load_config("irpd.json")["output_file_names"]



class FoundationalModel(ABC):
    """
    Base model for functional models.
    
    Has methods for subpath & meta generation, and getting subsets
    """
    def __init__(self, irpd_config: IRPDConfig):
        self.irpd_config = irpd_config
        self.case = irpd_config.case
        self.ra = irpd_config.ra
        self.treatment = irpd_config.treatment
        self.llms = irpd_config.llms
        self.llm_config = irpd_config.llm_config
        self.stages = irpd_config.stages
        self.cases = irpd_config.cases
        self.batches = irpd_config.batches
        self.test_path = Path(irpd_config.test_path)
        self.data_path = Path(irpd_config.data_path)
        self.prompts_path = Path(irpd_config.prompts_path)
        self.max_instances = irpd_config.max_instances
        self.total_replications = irpd_config.total_replications
        
        self.schemas = {
            stage: lazy_import("types.irpd_stage_schemas", f"Stage{stage}Schema")
            for stage in self.stages
        }
        self.subsets = {
            stage: self._get_subsets(stage)
            for stage in self.stages
        }
        
    def _generate_subpath(self, n: int, llm_str: str):
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
    
    def _get_subsets(self, stage_name: str):
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
    
    def _generate_meta_path(self, n: int, llm_str: str):
        """
        Generates the path for 'test' meta. File exists for each subpath.
        """
        subpath = self._generate_subpath(n, llm_str)
        return subpath / FILE_NAMES["meta"]