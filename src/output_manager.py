import logging
from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict
from models.base_model import RequestOut

log = logging.getLogger(__name__)


@dataclass
class StageRun:
    stage: str
    stage_outputs: Dict[str, List[RequestOut]] = field(default_factory=lambda: defaultdict(list))
    
    def get(self, instance_type: str):
        return self.stage_outputs[instance_type]
    
    def store(self, instance_type: str, output: RequestOut):
        self.stage_outputs[instance_type] += [output]
        log.info(f"STORED: Stored stage {self.stage}, instance {instance_type}.")


@dataclass
class TestRun:
    stage_runs: Dict[str, StageRun]
    n: int
    
    def get(self, stage: str, instance_type: str | None = None):
        if stage in self.stage_runs:
            if instance_type:
                return self.stage_runs[stage].get(instance_type)
            return self.stage_runs[stage].stage_outputs
        return None

    def store(self, stage_run: StageRun):
        self.stage_runs[stage_run.stage] = stage_run
    

@dataclass
class OutputManager:
    test_runs: Dict[str, Dict[int, TestRun]] = field(default_factory=dict)
    
    def get(self, test_id: str, n: int, stage: str = None, instance_type: str = None):
        if test_id in self.test_runs:
            test_run = self.test_runs[test_id][n]
            if stage:
                if instance_type:
                    return test_run.get(stage, instance_type)
                return test_run.get(stage)
            return test_run
        log.warning(f"`None` found for replicate {n} of test {test_id}.")
        return None
    
    def store(self, test_id: str, n: int, stage_run: StageRun):
        if test_id not in self.test_runs:
            self.test_runs[test_id] = {}
        
        if n not in self.test_runs[test_id]:
            new_test_run = TestRun(
                stage_runs={stage_run.stage: stage_run},
                n=n
            )
            self.test_runs[test_id][n] = new_test_run
        else:
            self.test_runs[test_id][n].store()
        
        log.info(f"STORED: Stored stage {stage_run.stage} in replicate {n} of test {test_id}")