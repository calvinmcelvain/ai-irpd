import logging
import pandas as pd
import time as t
from requests.exceptions import Timeout
from irpd.stages.base_stage import BaseStage
from utils import file_to_string, write_file
from irpd.output_manager import StageRun

log = logging.getLogger(__name__)


class Stage1(BaseStage):
    def __init__(self, test_config, sub_path, context, llm, max_instances, threshold):
        super().__init__(test_config, sub_path, context, llm, max_instances, threshold)
        self.stage = "1"
        self.schema = self.schemas[self.stage]
        self.output = StageRun(self.stage)
    
    def _get_system_prompt(self):
        system_prompts = {case: {} for case in self.cases}
        for c in self.cases:
            for i in self._get_instance_types(c):
                prompt_name = f"stg_1_{self.treatment}_{i}.md"
                prompt_path = self.prompt_path / c / self.ra / prompt_name
                system_prompts[c][i] =  file_to_string(prompt_path)
        self.system_prompts = system_prompts
        return None
    
    def _get_user_prompt(self):
        user_prompts = {case: {} for case in self.cases}
        for c in self.cases:
            for i in self._get_instance_types(c):
                df_name = f"{c}_{self.treatment}_{self.ra}_{i}.csv"
                df_path = self.data_path / "test" / df_name
                user_prompts[c][i] = pd.read_csv(df_path).to_dict("records")
        self.user_prompts = user_prompts
        return None
    
    def _process_output(self):
        meta_path = self.sub_path / "_test_info" / f"stg_{self.stage}_test_info.json"
        if not meta_path.exists():
            self._write_meta()
        
        text = f"# Stage {self.stage} Output Categories\n\n"
        for c in self.cases:
            pdf = True
            for i in self._get_instance_types(c):
                try:
                    output = self.output.get(c, i)[0]
                    text += self._output_to_txt(
                        output, self.schema, f"## {i.upper()} Categories\n\n"
                    )
                    
                    write_path = self.sub_path / f"stage_{self.stage}" / c / i
                    write_path.mkdir(exist_ok=True, parents=True)
                    prefix = f"stg_{self.stage}_{i}_"
                    system_path = write_path / (prefix + "sys_prmpt.txt")
                    user_path = write_path / (prefix + "user_prmpt.txt")
                    response_path = write_path / (prefix + "response.txt")
                    
                    if not any(path.exists() for path in [system_path, user_path, response_path]):
                        write_file(system_path, output.system)
                        write_file(user_path, output.user)
                        write_file(response_path, output.response)
                except Exception as e:
                    pdf = False
                    log.error(f"Error occured in processing {c}, instance {i}: {e}.")
                    continue
            pdf_path = self.sub_path / f"{c}_stg_{self.stage}_categories.pdf"
            if pdf:
                self._txt_to_pdf(text, pdf_path)
        
    def run(self):
        super().run()
        for c in self.cases:
            for i in self._get_instance_types(c):
                if not self._check_completed_requests(i, c):
                    retries = 0
                    while retries < self.retries:
                        try:
                            output = self.llm.request(
                                user=str(self.user_prompts[c][i]),
                                system=str(self.system_prompts[c][i]),
                                schema=self.schema
                            )
                            self.output.store(c, i, output)
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
            