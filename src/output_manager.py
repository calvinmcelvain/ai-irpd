import logging
from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict
from llms.base_model import RequestOut

log = logging.getLogger("app.output_manager")


@dataclass
class StageRun:
    stage: str
    stage_outputs: Dict[str, Dict[str, List[RequestOut]]] = field(default_factory=lambda: defaultdict(list))
    
    def get(self, case: str = None, instance_type: str = None):
        if case in self.stage_outputs:
            if instance_type in self.stage_outputs[case]:
                return self.stage_outputs[case][instance_type]
            return self.stage_outputs[case]
        return self.stage_outputs
    
    def has(self, case: str, instance_type: str):
        return case in self.stage_outputs and instance_type in self.stage_outputs[case]
    
    def store(self, case: str, instance_type: str, output: RequestOut):
        if case not in self.stage_outputs:
            self.stage_outputs[case] = {}
        if instance_type not in self.stage_outputs[case]:
            self.stage_outputs[case][instance_type] = []
        
        self.stage_outputs[case][instance_type] += [output]
        log.info(f"OUTPUTS: Stored stage {self.stage}, instance {instance_type}.")


@dataclass
class TestRun:
    n: int
    stage_runs: Dict[str, StageRun] = field(default_factory=dict)
    
    def get(self, stage: str, case: str = None, instance_type: str = None):
        if stage in self.stage_runs:
            if case in self.stage_runs[stage].stage_outputs:
                if instance_type in self.stage_runs[stage].stage_outputs[case]:
                    return self.stage_runs[stage].get(case, instance_type)
            return self.stage_runs[stage].get(case)
        return self.stage_runs
    
    def has(self, stage: str, case: str, instance_type: str):
        return stage in self.stage_runs and self.stage_runs[stage].has(case, instance_type)
    
    def store(self, stage_run: StageRun):
        self.stage_runs[stage_run.stage] = stage_run
    

@dataclass
class OutputManager:
    test_runs: Dict[str, Dict[int, TestRun]] = field(default_factory=dict)
    
    def get(
        self,
        test_id: str,
        n: int,
        stage: str = None,
        case: str = None,
        instance_type: str = None
    ):
        if test_id in self.test_runs:
            test_run = self.test_runs[test_id][n]
            if stage in test_run.stage_runs:
                if case in test_run.stage_runs[stage].stage_outputs:
                    stage_outputs = test_run.stage_runs[stage].stage_outputs
                    if instance_type in stage_outputs[case]:
                        return test_run.get(stage, case, instance_type)
                    return stage_outputs.get(stage, case)
                return test_run.get(stage)
            return test_run
        else:
            return TestRun(n)
    
    def store(self, test_id: str, n: int, stage_run: StageRun = None):
        if test_id not in self.test_runs:
            self.test_runs[test_id] = {}
        
        if n not in self.test_runs[test_id]:
            new_test_run = TestRun(
                stage_runs={stage_run.stage: stage_run},
                n=n
            )
            self.test_runs[test_id][n] = new_test_run
        else:
            self.test_runs[test_id][n].store(stage_run)
        
        log.info(f"OUTPUTS: Stored stage {stage_run.stage} in replicate {n} of test {test_id}")