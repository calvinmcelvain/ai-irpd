import logging
from pathlib import Path

from models.irpd.test_config import TestConfig
from models.irpd.test_prompts import TestPrompts
from models.irpd.outputs import TestOutput
from models.llm_model import LLMModel


log = logging.getLogger(__name__)



class TestRunner:
    def __init__(
        self,
        config: TestConfig,
        output: TestOutput,
        print_response: bool = False
    ):
        self.config = config
        self.case = config.case
        self.cases = config.case.split("_")
        self.stages = config.stages
        self.batch_request = config.batches
        self.llm_config = config.llm_config
        self.test_path = config.test_path
        self.total_replications = config.total_replications
        self.replications = config.total_replications
        self.llms = config.llms
        self.output = output
        self.print_response = print_response
    
    @staticmethod
    def _get_instance_types(case: str):
        if case in {"uni", "uniresp"}:
            return ["ucoop", "udef"]
        return ["coop", "def"]
    
    def _get_subsets(self, stage: str):
        if stage in {"1c", "2", "3"}:
            return ["full"]
        subsets = [
            f"{c}_{i}" for c in self.cases
            for i in self._get_instance_types(c)
        ]
        return subsets + ["full"]
        
    def _generate_subpath(self, N: int, llm_str: str):
        subpath = self.test_path
        if len(self.llms) > 1: subpath = subpath / llm_str
        if self.total_replications > 1: subpath = subpath / f"replication_{N}"
        return Path(subpath)
    
    def _generate_llm_instance(self, llm: str):
        return getattr(LLMModel, llm).get_llm_instance(
            self.llm_config, self.print_response
        )
        
    def _generate_prompts(self, stage: str, llm: str):
        subsets = self._get_subsets(stage)
        prompts = []
        prompt_ids = []
        for n in range(1, self.total_replications + 1):
            for subset in subsets:
                subpath = self._generate_subpath(n, llm)
                complete = self.output.check_output(subpath, llm, n, stage)
                if not complete:
                    context = self.output.retrieve(stage, llm, n, subset)
                    test_prompts = TestPrompts(stage, self.config, context)
                    # Need to fix TestPrompts so that the fixed argument is
                    # determined by the test type from the config passed on 
                    # initialization. Also, it should return a list already.
                    # A list for each user prompt. Zip list by id.
                    prompts.append(test_prompts.get_prompts(subset, self.case))
                    prompt_ids.append()
        return
    
    def _run_batch(self):
        # Need to be dynamic to some requests failing. Send new batch for failed
        # requests. 
        pass
    
    def _run_completions(self):
        # this function should be be dynamic to the replications and/or llms
        # that have already been run. check-in w/ the TestPrompts model &
        # either the TestOutput or Context models.
        return None
    
    def run(self):
        if self.batch_request:
            self._run_batch()
        else:
            self._run_completions()