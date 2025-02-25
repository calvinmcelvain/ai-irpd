import logging
from dataclasses import dataclass, field
from typing import Dict
from collections import defaultdict
from llms.base_model import RequestOut

log = logging.getLogger(__name__)


@dataclass
class StageRun:
    stage: str
    stage_outputs: defaultdict = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    
    def get(self, case: str = None, instance_type: str = None):
        if case:
            if instance_type:
                return self.stage_outputs[case][instance_type]
            return self.stage_outputs[case]
        return self.stage_outputs
    
    def has(self, case: str, instance_type: str):
        return instance_type in self.stage_outputs[case]
    
    def store(self, case: str, instance_type: str, output: RequestOut):
        self.stage_outputs[case][instance_type].append(output)
        log.info(
            f"OUTPUTS: Stored --Stage {self.stage} --Instance {instance_type}"
        )


@dataclass
class TestRun:
    n: int
    stage_runs: Dict[str, StageRun] = field(default_factory=dict)
    
    def get(self, stage: str, case: str = None, instance_type: str = None):
        if stage in self.stage_runs:
            return self.stage_runs[stage].get(case, instance_type)
        return self
    
    def has(self, stage: str, case: str, instance_type: str):
        return stage in self.stage_runs and self.stage_runs[stage].has(case, instance_type)
    
    def store(self, stage_run: StageRun):
        self.stage_runs[stage_run.stage] = stage_run
    

@dataclass
class OutputManager:
    test_runs: defaultdict = field(default_factory=lambda: defaultdict(lambda: defaultdict(dict)))
    
    def get(
        self,
        test_id: str,
        n: int,
        llm: str,
        stage: str = None,
        case: str = None,
        instance_type: str = None
    ):
        test_run = self.test_runs[test_id][llm].get(n)
        if test_run:
            return self.test_runs[test_id][llm][n].get(stage, case, instance_type)
        return TestRun(n)
    
    def store(self, test_id: str, n: int, llm: str, stage_run: StageRun = None):
        test_run = self.test_runs[test_id][llm].setdefault(n, TestRun(n=n))
        test_run.store(stage_run)
        log.info(
            f"OUTPUTS: Stored --Test {test_id} --Stage {stage_run.stage}"
            f" --Replicate {n} --LLM {llm} --Test {test_id}"
        )
        return None