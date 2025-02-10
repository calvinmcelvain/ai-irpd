import logging
from pathlib import Path
from itertools import product
from typing import List
from datetime import datetime
from abc import ABC, abstractmethod
from test_config import TestConfig
from output_manager import TestRun
from models.base_model import RequestOut
from utils import find_named_parent, validate_json_string, txt_to_pdf, write_json, load_json

log = logging.getLogger(__name__)


class BaseStage(ABC):
    def __init__(
        self,
        test_config: TestConfig,
        sub_path: Path,
        context: TestRun
    ):
        self.case = self._get_cases(test_config.case)
        self.ra = test_config.ra
        self.treatment = test_config.treatment
        self.llm = test_config.llm
        self.test_type = test_config.test_type
        self.test_path = test_config.test_path
        self.max_instances = test_config.max_instances
        self.project_path = find_named_parent(self.test_path, "output").parent
        self.prompt_path = self.project_path / "prompts"
        self.data_path = self.project_path / "data"
        self.sub_path = sub_path
        self.context = context
        self.instance_types = self._get_instance_types()
        self.product_ci = list(product(self.case, self.instance_types))
        
    def check_completed_requests(self, instance_type, case):
        log.info(
            f"CHECK: Checking for Stage {self.stage}, {case}, {instance_type} outputs."
        )
        name = f"stg_{self.stage}_{instance_type}_response.txt"
        path = self.sub_path / f"stage_{self.stage}" / case / instance_type / name
        if path.exists():
            log.info("CHECK: Outputs found.")
            response = load_json(path)
            self.output.store(case, instance_type, RequestOut(response=response))
            return True
        log.info("CHECK: Outputs not found.")
        return None
    
    def _get_instance_types(self):
        if any(c in {'uni', 'uniresp'} for c in self.case):
            return ['ucoop', 'udef']
        return ['coop', 'def']
    
    @staticmethod
    def _get_cases(case):
        if case == 'uni_switch':
            return ['uni', 'switch']
        return [case]
    
    @staticmethod
    def _format_categories(categories: List, initial_text: str = "") :
        category_texts = []
        for category in categories:
            examples_text = "\n".join(
                f"  {idx}. Window number: {example.window_number}, Reasoning: {example.reasoning}"
                for idx, example in enumerate(category.examples, start=1)
            )
            category_text = (
                f"### {category.category_name}\n\n"
                f"**Definition**: {category.definition}\n\n"
                f"**Examples**:\n\n{examples_text}\n\n"
            )
            category_texts.append(category_text)
        return initial_text + "".join(category_texts)
    
    def _output_to_pdf(self):
        for c in self.case:
            text = f"# {c.upper()} Stage {self.stage} Categories\n\n"
            
            for i in self.instance_types:
                output = self.output.get(c, i)[0]
                json_output = validate_json_string(output.response, self.schema)
                
                if json_output:
                    text += self._format_categories(
                        json_output.categories,
                        f"## {i.capitalize()} Categories\n\n"
                    )
            path = self.sub_path / f"{c}_stg_{self.stage}_categories.pdf"
            txt_to_pdf(text, path)
        return None
    
    def _compute_tokens(self):
        tokens = {case: {} for case in self.case}
        for case, instance_type in self.product_ci:
            outputs = self.output.get(case, instance_type)
            tokens[case][instance_type] = {
                "input_tokens": sum(output.meta.usage.prompt_tokens for output in outputs),
                "output_tokens": sum(output.meta.usage.completion_tokens for output in outputs),
                "total_tokens": sum(output.meta.usage.total_tokens for output in outputs)
            }
        tokens["total"] = {
            "input_tokens": sum(tokens[c][i]["input_tokens"] for c, i in self.product_ci),
            "output_tokens": sum(tokens[c][i]["output_tokens"] for c, i in self.product_ci),
            "total_tokens": sum(tokens[c][i]["total_tokens"] for c, i in self.product_ci)
        }
        return tokens
    
    def _write_meta(self):
        model = self.llm.model
        parameters = self.llm.configs
        created = str(datetime.now())
        try:
            tokens = self._compute_tokens()
        except AttributeError:
            tokens = None
        
        meta_path = self.sub_path / "_test_info"
        meta_path.mkdir(exist_ok=True)
        
        json_data = {
            "created": created,
            "model_info": {
                "model": model,
                "parameters": parameters.model_dump()
            },
            "test_info": {
                "case_types": self.case,
                "ra": self.ra,
                "treatment": self.treatment,
                "test_type": self.test_type,
                "test_path": self.test_path.as_posix()
            },
            "tokens": tokens
        }
        
        write_json(meta_path / f"stg_{self.stage}_test_info.json", json_data)
        
        return None
    
    @abstractmethod
    def check_completed_requests(self):
        pass
    
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