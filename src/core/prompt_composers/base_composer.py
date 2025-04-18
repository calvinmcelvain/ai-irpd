"""
Contains the BaseComposer model.
"""
import logging
from pathlib import Path
from abc import abstractmethod
from typing import List, Tuple, Dict

from helpers.utils import file_to_string, load_config
from core.foundation import FoundationalModel
from core.data import Data
from _types.prompts import Prompts
from _types.irpd_config import IRPDConfig
from _types.test_output import TestOutput


log = logging.getLogger("app")


CONFIGS = load_config("irpd.json")



class BaseComposer(FoundationalModel):
    """
    BaseComposer model.
    
    The base model for stage-specific composers.
    """
    def __init__(self, irpd_config: IRPDConfig, stage_name: str):
        super().__init__(irpd_config)
        
        self.stage_name = stage_name
        self.data = Data(irpd_config)
        
        # Categories are fixed for stages 2 & 3 if a 'replication' test type.
        self.fixed = self.test_type in CONFIGS["test_types"]["class"]["replication"]

        self.sections_path = Path(self.prompts_path / "sections")
        self.fixed_path = Path(self.prompts_path / "fixed")
        
    @staticmethod
    def _get_section(section_path: Path, name: str) -> str:
        """
        Returns the prompt section & logs if was found to be empty.
        """
        section = file_to_string(section_path)
        if not section:
            log.warning(f"{name} was empty.")
        return section + "\n"
    
    @staticmethod
    def _prompt_id(stage_name: str, subset: str, n: int, user: dict) -> str:
        """
        Generates the prompt ID.
        """
        prompt_id = f"{n}-{subset}"
        if stage_name in CONFIGS["stage_class"]["classification"]:
            prompt_id += f"-{user["window_number"]}"
        return prompt_id
    
    @abstractmethod
    def _task_overview(self) -> str:
        """
        Returns the 'Task Overview' section of system prompt.
        """
        section_directory = self.sections_path / "task_overview"
        section_path = section_directory / f"stage_{self.stage_name}.md"
        return self._get_section(section_path, "Task Overview")
    
    @abstractmethod
    def _experimental_context(self) -> str:
        """
        Returns the 'Experimental Context' section of the system prompt.
        """
        section_directory = self.sections_path / "experimental_context"
        section_path = section_directory / f"{self.treatment}.md"
        return self._get_section(section_path, "Experimental Context")
    
    @abstractmethod
    def _summary_context(self) -> str:
        """
        Returns the 'Summary Context' section of the system prompt.
        """
        section_directory = self.sections_path / "summary_context"
        section_path = section_directory / f"{self.case}_{self.ra}.md"
        return self._get_section(section_path, "Summary Context")
    
    @abstractmethod
    def _task(self) -> str:
        """
        Returns the stasge task section(s) of the system prompt.
        
        Essentially just the instructions for the stage.
        """
        section_directory = self.sections_path / "task"
        section_path = section_directory / f"stage_{self.stage_name}.md"
        return self._get_section(section_path, "Task")
    
    @abstractmethod
    def _constraints(self) -> str:
        """
        Returns the 'Constraints' section of the system prompt.
        
        Apart of the stage specific instructions.
        """
        section_directory = self.sections_path / "constraints"
        section_path = section_directory / f"stage_{self.stage_name}.md"
        return self._get_section(section_path, "Constraints")
    
    @abstractmethod
    def _data_definitions(self) -> str:
        """
        Returns the 'Data Variable Definitions' section of the system prompt.
        """
        section_directory = self.sections_path / "data_definitions"
        
        # Initial prompt section (header).
        section = file_to_string(section_directory / "initial.md")
        
        # The rest is filled by child classes.
        return section
    
    @abstractmethod
    def _construct_system_prompt(self) -> str | List[Dict] | List[List[Dict]]:
        """
        Constructs the system prompt.
        """
        pass
    
    @abstractmethod
    def expected_outputs(self) -> int:
        """
        Returns the expected number of outputs for a given stage.
        """
        pass
    
    @abstractmethod
    def get_prompts(
        self, test_outputs: List[TestOutput]
    ) -> List[Tuple[str, Prompts]]:
        """
        Returns a list of tuples w/ first element corresponding to the prompt ID
        and the second corresponding to a Prompts object.
        """
        pass