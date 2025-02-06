from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict
from models.base_model import RequestOut
from test_config import TestConfig


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
    test_config: TestConfig
    
    def get(self, stage: str, instance_type: str | None = None):
        if stage in self.stage_runs:
            if instance_type:
                return self.stage_runs[stage].get(instance_type)
            return self.stage_runs[stage].stage_outputs
        return None

    def store(self, stage: str, output: RequestOut, instance_type: str):
        if stage not in self.stage_runs:
            self.stage_runs[stage] = StageRun(stage=stage)
        self.stage_runs[stage].store(instance_type, output)
    

@dataclass
class OutputManager:
    test_runs: Dict[str, List[TestRun]] = field(default_factory=dict)
    test_configs: Dict[str, TestConfig] = field(default_factory=dict)
    
    def get(self, test_id: str, stage: str, instance_type: str = None):
        if test_id in self.test_runs:
            for test_run in self.test_runs[test_id]:
                result = test_run.get(stage, instance_type)
                if result:
                    return result
        return None
    
    def store(
        self, test_id: str, stage: str, output: RequestOut, instance_type: str
    ):
        if test_id not in self.test_runs:
            self.test_runs[test_id] = []
        
        if not any(stage in test_run.stage_runs for test_run in self.test_runs[test_id]):
            test_config = self.test_configs.get(test_id)
            new_test_run = TestRun(
                test_id=test_id,
                stage_runs={stage: StageRun(stage=stage)},
                test_config=test_config
            )
            self.test_runs[test_id].append(new_test_run)

        for test_run in self.test_runs[test_id]:
            if stage in test_run.stage_runs:
                test_run.store(stage, output, instance_type)
                break
    
    def get_config(self, test_id: str):
        return self.test_configs.get(test_id, None)

    def add_config(self, test_config: TestConfig):
        self.test_configs[test_config.id] = test_config