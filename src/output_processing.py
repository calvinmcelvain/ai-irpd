from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict
from models.base_model import RequestOut


@dataclass
class StageRun:
    stage: str
    stage_outputs: Dict[str, List[RequestOut]] = field(default_factory=lambda: defaultdict(list))
    
    def get(self, instance_type: str):
        return self.stage_outputs[instance_type]
    
    def store(self, instance_type: str, output: RequestOut):
        self.stage_outputs[instance_type] += [output]


@dataclass
class TestRun:
    stage_runs: Dict[str, StageRun]
    test_id: str
    
    def get(self, stage: str, instance_type: str | None = None):
        if instance_type:
            return self.stage_runs[stage].get(instance_type)
        else:
            return self.stage_runs[stage].stage_outputs
    
    def store(self, stage: str, output: RequestOut, instance_type: str):
        self.stage_runs[stage].store(instance_type, output)
    

@dataclass
class OutputManager:
    test_runs: Dict[str, TestRun] = field(default_factory=dict)
    
    def get(self, test_id: str, stage: str, instance_type: str = None):
        if test_id in self.test_runs:
            return self.test_runs[test_id].get(stage, instance_type)
        return None
    
    def store(self, test_id: str, stage: str, output: RequestOut, instance_type: str):
        if test_id not in self.test_runs:
            self.test_runs[test_id] = TestRun(stage_runs={}, test_id=test_id)
        if stage not in self.test_runs[test_id].stage_runs:
            self.test_runs[test_id].stage_runs[stage] = StageRun(stage=stage)
        self.test_runs[test_id].store(stage, output, instance_type)