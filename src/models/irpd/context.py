from pathlib import Path
from dataclasses import dataclass

from models.irpd.outputs import TestOutput


@dataclass
class Context:
    llm_str: str
    replication: int
    sub_path: Path
    context: TestOutput
    
    def update(self, stage_name: str):
        found = self.context.check_output(
            self.sub_path, self.llm_str, self.replication, stage_name
        )
        return found
        
