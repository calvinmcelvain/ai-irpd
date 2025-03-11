import logging
from pathlib import Path
import time as t
from requests.exceptions import Timeout

from utils import txt_to_pdf
from models.irpd.stages.base_stage import BaseStage
from models.irpd.test_config import TestConfig
from models.irpd.test_prompts import TestPrompts
from models.llms.base_llm import BaseLLM
from models.irpd.stage_output import StageOutput


log = logging.getLogger(__name__)



class Stage1(BaseStage):
    def __init__(
        self,
        test_config: TestConfig,
        sub_path: Path,
        prompts: TestPrompts,
        llm: BaseLLM
    ):
        super().__init__(test_config, sub_path, prompts, llm)
        self.stage = "1"
        self.output = StageOutput(stage=self.stage)
        self.schema = self._get_stage_schema()
    
    def _process_output(self):
        self._write_meta()
        
        pdf = "# Stage 1 Categories\n\n"
        for subset in self.subsets:
            if subset in self.output.outputs.keys():
                if len(subset.split("_")) == 2:
                    case, sub = subset.split("_")
                    pdf += f"## {case.capitalize()}; {sub.upper()} Categories\n\n"
                    categories = self._get_att(self.output.outputs[subset].parsed)
                    pdf += self._categories_to_txt(categories=categories)
                else:
                    pdf += f"## Unified Categories\n\n"
                    categories = self._get_att(self.output.outputs[subset].parsed)
                    pdf += self._categories_to_txt(categories=categories)
        txt_to_pdf(pdf)
        return None

    def run(self):
        for subset in self.subsets:
            retries = 0
            if not self._check_context(subset=subset):
                prompts = self.prompts.get_prompts(subset=subset, case=self.case, fixed=self.fixed)
                while retries < self.retries:
                    try:
                        output = self.llm.request(
                            user=str(prompts.user),
                            system=str(prompts.system),
                            schema=self.schema
                        )
                        self.output.outputs[subset] = output
                        break
                    except Timeout:
                        retries += 1
                        log.warning("HTTP Request Timeout. Retrying...")
                        t.sleep(3)
                    except Exception as e:
                        log.error(f"Stage 1 error: {e}")
                        self._process_output()
                        raise Exception
        if retries == self.retries:
            log.error("Max retries for HTTP requests was hit.")
        self._process_output()
            