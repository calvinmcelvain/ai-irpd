import logging
from pathlib import Path
from utils import get_env_var, file_to_string

log = logging.getLogger(__name__)


class Prompt:
    def __init__(
        self,
        case: str,
        stage: str,
        treatmnet: str,
        ra: str,
        test_type: str,
        prompt_path: str | Path = None
    ):
        self.case = case
        self.stage = stage
        self.treatment = treatmnet
        self.ra = ra
        self.test_type = test_type
        
        if not prompt_path:
            prompt_path = Path(get_env_var("PROMPT_DIRECTORY"))
        if isinstance(prompt_path, str):
            prompt_path = Path(prompt_path)
        self.sections_path = prompt_path / "sections"
        self.fixed_path = prompt_path / "fixed"
    
    @staticmethod
    def _get_section(section_path, name):
        section = file_to_string(section_path)
        if not section:
            log.warning(f"PROMPTS: {name} was empty.")
        return section
    
    def _task_overview(self):
        section_path = self.sections_path / "task_overview" / f"stage_{self.stage}.md"
        return self._get_section(section_path, "Task Overview")

    def _experimental_context(self):
        section_path = self.sections_path / "experimental_context" / f"{self.treatment}.md"
        return self._get_section(section_path, "Experimental Context")
    
    def _summary_context(self):
        section_path = self.sections_path / "summary_context" / f"{self.case}_{self.ra}.md"
        return self._get_section(section_path, "Summary Context")
    
    def _task(self):
        section_path = self.sections_path / "task" / f"stage_{self.stage}.md"
        return self._get_section(section_path, "Task")
    
    def _constrains(self, type: str = "category_name"):
        section_path = self.sections_path / "constraints" / f"{type}.md"
        return self._get_section(section_path, "Constraints")
    
    def _data_definitions(self):
        section_path = self.sections_path / "data_definitions"
        initial = file_to_string(section_path / "initial.md")
        ra = file_to_string(section_path / f"{self.ra}.md")
        window = file_to_string(section_path / "window_number.md")
        section = initial + ra + window
        if self.stage == "1c":
            section += file_to_string(section_path / "instance_types" / f"{self.case}.md")
        if not section:
            log.warning(f"PROMPTS: Data Definitions was empty.")
            return section
        return section + "\n"
    
    def construct_prompt(self):
        pass
        
        