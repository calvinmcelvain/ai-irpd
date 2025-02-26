import logging
import pandas as pd
import time as t
from requests.exceptions import Timeout
from llms.base_model import RequestOut
from irpd.stages.base_stage import BaseStage
from utils import file_to_string, write_file, validate_json_string
from irpd.output_manager import StageRun

log = logging.getLogger(__name__)


class Stage1c(BaseStage):
    def __init__(self, test_config, sub_path, context, llm, max_instances, threshold):
        super().__init__(test_config, sub_path, context, llm, max_instances, threshold)
        self.stage = "1c"
        self.output = StageRun(self.stage)
        self.parts = ["part_1", "part_2"]
        self.schema_map = {"part_1": "1", "part_2": "1r"}
    
    def _get_system_prompt(self):
        path = self.prompt_path / self.case / self.ra
        if self.output.has(self.case, "part_1"):
            return file_to_string(path / f"stg_1r_{self.treatment}.md")
        return file_to_string(path / f"stg_1c_{self.treatment}.md")
    
    def _get_user_prompt(self):
        if self.output.has(self.case, "part_1"):
            output = self.output.get(self.case, "part_1")
            return self._output_to_txt(
                output[0], self.schemas[self.schema_map["part_1"]]
            )
        df_name = f"{self.case}_{self.treatment}_{self.ra}.csv"
        df_path = self.data_path / "test" / df_name
        return pd.read_csv(df_path).to_dict("records")

    def _compute_tokens(self):
        tokens = {self.case: {}}
        for i in self.parts:
            outputs = self.output.get(self.case, i)
            tokens[self.case][i] = {
                "input_tokens": sum(output.meta.usage.prompt_tokens for output in outputs),
                "output_tokens": sum(output.meta.usage.completion_tokens for output in outputs),
                "total_tokens": sum(output.meta.usage.total_tokens for output in outputs)
            }
        tokens["total"] = {
            "input_tokens": sum(tokens[self.case][i]["input_tokens"] for i in self.parts),
            "output_tokens": sum(tokens[self.case][i]["output_tokens"] for i in self.parts),
            "total_tokens": sum(tokens[self.case][i]["total_tokens"] for i in self.parts)
        }
        return tokens
    
    def _process_output(self):
        meta_path = self.sub_path / "_test_info" / f"stg_{self.stage}_test_info.json"
        if not meta_path.exists():
            self._write_meta()
        
        for part in self.parts:
            output = self.output.get(self.case, part)[0]
              
            write_path = self.sub_path / f"stage_{self.stage}" / part
            write_path.mkdir(exist_ok=True, parents=True)
            prefix = f"stg_{self.stage}_{part}_"
            system_path = write_path / (prefix + "sys_prmpt.txt")
            user_path = write_path / (prefix + "user_prmpt.txt")
            response_path = write_path / (prefix + "response.txt")
            
            if not any(path.exists() for path in [system_path, user_path, response_path]):
                write_file(system_path, output.system)
                write_file(user_path, output.user)
                write_file(response_path, output.response)
        
        text = f"# Stage {self.stage} Output Categories\n\n"
        pt2_output = self.output.get(self.case, "part_2")[0]
        text += self._output_to_txt(
            pt2_output, self.schemas[self.schema_map["part_2"]], f"## Unified Categories\n\n"
        )
        pdf = True
        for c in self.cases:
            for i in self._get_instance_types(c):
                try:
                    if not self.context.has("1r", c, i):
                        self._update_context("1r", c)
                    output = self.context.get("1r", c, i)[0]
                    categories = validate_json_string(
                        output.response, self.schemas["1r"]
                    )
                    ucategories = validate_json_string(
                        pt2_output.response, self.schemas["1r"]
                    )
                    
                    new_1r_cats = self._threshold_similarity(
                        categories, ucategories
                    )
                    output = RequestOut(response = new_1r_cats.model_dump_json())
                    text += self._output_to_txt(
                        output,
                        self.schemas["1r"],
                        f"## {i.upper()} Categories\n\n"
                    )
                except Exception as e:
                    pdf = False
                    log.error(f"Error occured in processing {c}, part {i}: {e}.")
                    continue
        pdf_path = self.sub_path / f"stg_{self.stage}_categories.pdf"
        if pdf:
            self._txt_to_pdf(text, pdf_path)
        
    def run(self):
        for part in self.parts:
            if not self._check_completed_requests(part, self.case):
                retries = 0
                while retries < self.retries:
                    try:
                        system_prompt = self._get_system_prompt()
                        user_prompt = self._get_user_prompt()
                        
                        output = self.llm.request(
                            user=str(user_prompt),
                            system=str(system_prompt),
                            schema=self.schemas[self.schema_map[part]]
                        )
                        self.output.store(self.case, part, output)
                        break
                    except Timeout:
                        retries += 1
                        log.warning("HTTP Request Timeout. Retrying...")
                        t.sleep(3)
                    except Exception as e:
                        log.error(f"Stage 1 error: {e}")
                        self._process_output()
                        raise Exception
        if self.retries == 3:
            log.error("Max retries for HTTP requests was hit.")
        self._process_output()
            
            