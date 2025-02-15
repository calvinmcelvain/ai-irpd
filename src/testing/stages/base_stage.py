import logging
from pathlib import Path
from typing import List
from datetime import datetime
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from test_config import TestConfig
from output_manager import TestRun, StageRun
from llms.base_model import RequestOut
from testing.stages.schemas import (
    Stage0Schema, Stage1Schema, Stage1rSchema, Stage2Schema, Stage3Schema
)
from utils import (
    find_named_parent, validate_json_string, txt_to_pdf, write_json, load_json
)

log = logging.getLogger("app.base_stage")


class BaseStage(ABC):
    schemas = {
        "0": Stage0Schema,
        "1": Stage1Schema,
        "1r": Stage1rSchema,
        "2": Stage2Schema,
        "3": Stage3Schema
    }
    
    def __init__(
        self,
        test_config: TestConfig,
        sub_path: Path,
        context: TestRun,
        max_instances: int | None,
        threshold: float
    ):
        self.case = test_config.case
        self.cases = self._get_cases(self.case)
        self.ra = test_config.ra
        self.treatment = test_config.treatment
        self.llm = test_config.llm
        self.test_type = test_config.test_type
        self.test_path = test_config.test_path
        self.max_instances = max_instances
        self.project_path = find_named_parent(self.test_path, "output").parent
        self.prompt_path = self.project_path / "prompts"
        self.data_path = self.project_path / "data"
        self.sub_path = sub_path
        self.context = context
        self.threshold = threshold
        
    def _check_completed_requests(self, instance_type, case):
        if not self.context.has(self.stage, case, instance_type):
            log.info(
                f"OUTPUTS: Checking for Stage {self.stage}, {case}, {instance_type} outputs."
            )
            name = f"stg_{self.stage}_{instance_type}_response.txt"
            path = self.sub_path / case / f"stage_{self.stage}" / instance_type / name
            if path.exists():
                log.info("OUTPUTS: Outputs found.")
                response = load_json(path, True)
                self.output.store(case, instance_type, RequestOut(response=response))
                return True
            log.info("OUTPUTS: Outputs not found.")
            return None
        return None
    
    def _update_context(self, stage, case):
        instance_types = self._get_instance_types(case)
        for i in instance_types:
            if not self.context.has(self.stage, case, i):
                log.info(
                    f"OUTPUTS: Outputs for Stage {stage}, {case}, {i} not found in context."
                    " Checking test path."
                )
        log.info(f"OUTPUTS: Getting outputs for Stage {stage}, case {case}.")
        stage_run = StageRun(stage)
        for i in instance_types:
            path = self.sub_path / case / f"stage_{stage}" / i / f"stg_{stage}_{i}_response.txt"
            if path.exists():
                log.info("OUTPUTS: Outputs retreived.")
                response = load_json(path, True)
                stage_run.store(case, i, RequestOut(response=response))
        self.context.store(stage_run)
        log.info(f"OUTPUTS: Stage {stage}, case {case} outputs stored in context.")
        return None
    
    @staticmethod
    def _get_instance_types(case):
        if case in {'uni', 'uniresp'}:
            return ['ucoop', 'udef']
        return ['coop', 'def']
    
    @staticmethod
    def _get_cases(case):
        if case == 'uni_switch':
            return ['uni', 'switch']
        return [case]
    
    @staticmethod
    def _format_categories(categories: List, initial_text: str = ""):
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
    
    def threshold_similarity(self, categories: List, unified_categories: List):
        all_cat_names = [
            cat.category_name.replace("_", " ")
            for cat in categories + unified_categories
        ]
        all_cat_defs = [
            cat.definition for cat in categories + unified_categories
        ]
        cat_names_w_ids = list(enumerate(all_cat_names[:len(categories)]))
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix_names = vectorizer.fit_transform(all_cat_names)
        tfidf_matrix_defs = vectorizer.fit_transform(all_cat_defs)
        
        num_categories = len(cat_names_w_ids)
        
        sim_matrix_names = cosine_similarity(
            tfidf_matrix_names[:num_categories], 
            tfidf_matrix_names[num_categories:]
        )
        sim_matrix_defs = cosine_similarity(
            tfidf_matrix_defs[:num_categories], 
            tfidf_matrix_defs[num_categories:]
        )
        sim_matrix = sim_matrix_names + sim_matrix_defs
        
        results = []
        for cat_idx, sim_array in enumerate(sim_matrix):
            if all(sim_array < self.threshold):
                results.append(categories[cat_idx])
        return results
    
    @staticmethod
    def _get_category_att(output):
        if hasattr(output, "categories"):
            return output.categories
        if hasattr(output, "refined_categories"):
            return output.refined_categories
    
    def _output_to_pdf(self):
        for c in self.cases:
            text = f"# {c.upper()} Stage {self.stage} Categories\n\n"
            
            for i in self._get_instance_types(c):
                output = self.output.get(c, i)[0]
                json_output = validate_json_string(output.response, self.schema)
                
                if json_output:
                    categories = self._get_category_att(json_output)
                    text += self._format_categories(
                        categories,
                        f"## {i.capitalize()} Categories\n\n"
                    )
            path = self.sub_path / f"{c}_stg_{self.stage}_categories.pdf"
            txt_to_pdf(text, path)
        return None
    
    def _output_to_txt(self, output, output_schema):
        txt = ""
        json_output = validate_json_string(output.response, output_schema)
        if json_output:
            categories = self._get_category_att(json_output)
            txt += self._format_categories(categories)
        return txt
    
    def _compute_tokens(self):
        tokens = {case: {} for case in self.cases}
        for c in self.cases:
            for i in self._get_instance_types(c):
                outputs = self.output.get(c, i)
                tokens[c][i] = {
                    "input_tokens": sum(output.meta.usage.prompt_tokens for output in outputs),
                    "output_tokens": sum(output.meta.usage.completion_tokens for output in outputs),
                    "total_tokens": sum(output.meta.usage.total_tokens for output in outputs)
                }
        tokens["total"] = {
            "input_tokens": sum(
                tokens[c][i]["input_tokens"] for c in self.cases for i in self._get_instance_types(c)),
            "output_tokens": sum(tokens[c][i]["output_tokens"] for c in self.cases for i in self._get_instance_types(c)),
            "total_tokens": sum(tokens[c][i]["total_tokens"] for c in self.cases for i in self._get_instance_types(c))
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
                "case": self.case,
                "ra": self.ra,
                "treatment": self.treatment,
                "test_type": self.test_type,
                "test_path": self.test_path.relative_to(self.project_path).as_posix()
            },
            "tokens": tokens
        }
        
        write_json(meta_path / f"stg_{self.stage}_test_info.json", json_data)
        
        return None
    
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