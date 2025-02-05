import logging
from pathlib import Path
from test_config import TestConfig
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class BaseStage(ABC):
    def __init__(
        self,
        test_config: TestConfig,
        outpath: Path
    ):
        self.case = self._get_cases(test_config.case)
        self.stage = None
        self.ra = test_config.ra
        self.treatment = test_config.treatment
        self.test_type = test_config.test_type
        self.test_path = test_config.test_path
        self.project_path = test_config.project_path
        self.prompt_path = self.project_path / "prompts"
        self.test_data_path = self.project_path / "data" / "test"
        self.raw_data_path = self.project_path / "data" / "raw"
        self.out_path = outpath
        self.llm = test_config.llm_instance
        self.max_instances = test_config.max_instances
        self.print_response = test_config.print_response
        self.instance_types = self._get_instance_types()
    
    def _get_instance_types(self):
        if self.case in ['uni', 'uniresp']:
            instance_types = ['ucoop', 'udef']
        elif self.case in ['switch', 'first']:
            instance_types = ['coop', 'def']
        return instance_types
    
    @staticmethod
    def _get_cases(case):
        if case == 'uni_switch':
            return ['uni', 'switch']
        return [case]
    
    @abstractmethod
    def _get_system_prompt(self):
        pass
    
    @abstractmethod
    def _get_user_prompt(self):
        pass
    
    @abstractmethod
    def _process_output(self):
        pass
    
    @abstractmethod
    def run(self):
        pass