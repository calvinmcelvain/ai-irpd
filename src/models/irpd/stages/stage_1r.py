import logging
from pathlib import Path
import time as t
from requests.exceptions import Timeout

from models.irpd.stages.base_stage import BaseStage
from models.irpd.test_config import TestConfig
from models.irpd.test_prompts import TestPrompts
from models.irpd.test_output import TestOutput
from models.llms.base_llm import BaseLLM


log = logging.getLogger(__name__)



class Stage1r(BaseStage):
    def __init__(
        self,
        test_config: TestConfig,
        sub_path: Path,
        prompts: TestPrompts,
        context: TestOutput,
        llm: BaseLLM,
        **kwargs
    ):
        self.stage = "1r"
        super().__init__(test_config, sub_path, prompts, context, llm, **kwargs)
    
    def _process_output(self):
        self._write_meta()
        self._build_categories_pdf()
        return None
        
    def run(self):
        for subset in self.subsets:
            self.output.outputs[subset] = []
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
                        self.output.outputs[subset] += [output]
                        break
                    except Timeout:
                        retries += 1
                        log.warning("HTTP Request Timeout. Retrying...")
                        t.sleep(3)
                    except Exception as e:
                        log.error(f"Stage 1r error: {e}")
                        self._process_output()
                        raise Exception
        if retries == self.retries:
            log.error("Max retries for HTTP requests was hit.")
        self._process_output()
            