import logging
import pandas as pd
import time as t
from requests.exceptions import Timeout
from irpd.stages.base_stage import BaseStage
from utils import file_to_string, write_file, load_json, validate_json_string
from irpd.output_manager import StageRun
from llms.base_model import RequestOut

log = logging.getLogger(__name__)


class Stage3(BaseStage):
    def __init__(self, test_config, sub_path, context, llm, max_instances, threshold):
        super().__init__(test_config, sub_path, context, llm, max_instances, threshold)
        self.stage = "3"
        self.schema = self.schemas[self.stage]
        self.output = StageRun(self.stage)
    
    def _check_completed_requests(self, instance_type, case):
        if not self.context.has(self.stage, case, instance_type):
            log.info(
                f"OUTPUTS: Checking for Stage {self.stage}, {case}, {instance_type} outputs."
            )
            path = self.sub_path / f"stage_{self.stage}" / case / instance_type / "responses"
            if path.exists():
                log.info("OUTPUTS: Outputs found.")
                for response in path.iterdir():
                    if response.name.endswith("response.txt"):
                        json_response = load_json(response, True)
                        self.output.store(case, instance_type, RequestOut(response=json_response))
                return True
            log.info("OUTPUTS: Outputs not found.")
            return None
        return True
    
    def _get_stg2(self, case):
        instance_types = self._get_instance_types(case)
        for i in instance_types:
            if not self.context.has("2", case, i):
                log.info(
                    f"OUTPUTS: Outputs for Stage 2, {case}, {i} not found in context."
                    " Checking test path."
                )
        log.info(f"OUTPUTS: Getting outputs for Stage 2, case {case}.")
        stage_run = StageRun("2")
        for i in instance_types:
            path = self.sub_path / "stage_2" / case / i / "responses"
            for response in path.iterdir():
                    if response.name.endswith("response.txt"):
                        response = load_json(response, True)
                        stage_run.store(case, i, RequestOut(response=response))
        self.context.store(stage_run)
        log.info(f"OUTPUTS: Stage 2, case {case} outputs stored in context.")
        return None
    
    def _get_stg1c_pt2(self):
        log.info(f"OUTPUTS: Getting outputs for Stage 1c part 2.")
        stage_run = StageRun("1c")
        path = self.sub_path / f"stage_1c" / "part_2" / f"stg_1c_part_2_response.txt"
        if path.exists():
            log.info("OUTPUTS: Outputs retreived.")
            response = load_json(path, True)
            stage_run.store(self.case, "part_2", RequestOut(response=response))
            self.context.store(stage_run)
            log.info(f"OUTPUTS: Stage 1c part 2 outputs stored in context.")
        else:
            log.warning("OUTPUTS: Stage 1c part 2 not found. Proceeding without.")
        return None
    
    def _get_system_prompt(self):
        text = ""
        if not self.context.has("1c", self.case, "part_2"):
            self._get_stg1c_pt2()
        if self.context.has("1c", self.case, "part_2"):
            pt2_output = self.context.get("1c", self.case, "part_2")[0]
            text 
            text += "\n" + self._output_to_txt(pt2_output, self.schemas["1r"])
        
        system_prompts = {case: {} for case in self.cases}
        for c in self.cases:
            for i in self._get_instance_types(c):
                prompt_name = f"stg_3_{self.treatment}.md"
                prompt_path = self.prompt_path / c / self.ra / prompt_name
                system_prompts[c][i] = file_to_string(prompt_path)
                
                if not self.context.has("1r", c, i):
                    self._update_context("1r", c)
                stg1r = self.context.get("1r", c, i)[0]
                
                if text:
                    system_prompts[c][i] += text
                    categories = validate_json_string(
                        stg1r.response, self.schemas["1r"]
                    )
                    ucategories = validate_json_string(
                        pt2_output.response, self.schemas["1r"]
                    )
                    
                    new_1r_cats = self._threshold_similarity(
                        categories, ucategories
                    )
                    output = RequestOut(response=new_1r_cats.model_dump_json())
                system_prompts[c][i] += self._output_to_txt(output, self.schemas["1r"])
        self.system_prompts = system_prompts
        return None
    
    def _get_user_prompt(self):
        user_prompts = {case: {} for case in self.cases}
        for c in self.cases:
            self._get_stg2(c)
            df_name = f"{c}_stg_2_final_output.csv"
            df = pd.read_csv(self.sub_path / df_name)
            for i in self._get_instance_types(c):
                if self._check_completed_requests(i, c):
                    responses = [
                        validate_json_string(r.response, self.schema)
                        for r in self.output.get(c, i)
                    ]
                    window_nums = [r.window_number for r in responses]
                else:
                    window_nums = []
                outputs = self.context.get("2", c, i)
                response_list = []
                for output in outputs:
                    json_out = validate_json_string(output.response, self.schemas["2"])
                    summary_df = df[df.columns.intersection(["summary_1", "summary_2", "window_number"])]
                    if not json_out.window_number in window_nums:
                        summary_df = summary_df[summary_df["window_number"] == json_out.window_number]
                        response = pd.DataFrame(summary_df).to_dict("records")[0]
                        response["assigned_categories"] = [cat.category_name for cat in json_out.assigned_categories]
                        response_list.append(response)
                user_prompts[c][i] = response_list
        self.user_prompts = user_prompts
        return None
    
    def _process_output(self):
        meta_path = self.sub_path / "_test_info" / f"stg_{self.stage}_test_info.json"
        if not meta_path.exists():
            self._write_meta()
        
        for c in self.cases:
            create_df = True
            for i in self._get_instance_types(c):
                try:
                    write_path = self.sub_path / f"stage_{self.stage}" / c / i 
                    write_path.mkdir(exist_ok=True, parents=True)
                    prefix = f"stg_{self.stage}_{i}_"
                    system_path = write_path / (prefix + "sys_prmpt.txt")
                    if not system_path.exists():
                        write_file(system_path, self.system_prompts[c][i])
                    
                    prompts_path = write_path / "prompts"
                    responses_path = write_path / "responses"
                    prompts_path.mkdir(exist_ok=True)
                    responses_path.mkdir(exist_ok=True)
                    for response in self.output.get(c, i):
                        output = validate_json_string(response.response, self.schema)
                        prefix = f"{output.window_number}_"
                        user_path = prompts_path / (prefix + "prompt.txt")
                        response_path = responses_path / (prefix + "response.txt")
                        
                        if not any(path.exists() for path in [user_path, response_path]):
                            write_file(user_path, response.user)
                            write_file(response_path, response.response)
                except Exception as e:
                    create_df = False
                    log.error(f"Error occured in processing {c}, instance {i}: {e}.")
                    continue
            df_path = self.sub_path / f"{c}_stg_{self.stage}_final_output.csv"
            if create_df:
                df = self._build_data_output(c)
                df.to_csv(df_path, index=False)
        
    def run(self):
        super().run()
        for c in self.cases:
            for i in self._get_instance_types(c):
                if len(self.user_prompts[c][i]) != 0:
                    for row in self.user_prompts[c][i]:
                        retries = 0
                        while retries < self.retries:
                            try:
                                output = self.llm.request(
                                    user=str(row),
                                    system=str(self.system_prompts[c][i]),
                                    schema=self.schema
                                )
                                self.output.store(c, i, output)
                            except Timeout:
                                retries += 1
                                log.warning("HTTP Request Timeout. Retrying...")
                                t.sleep(3)
                            except Exception as e:
                                log.error(f"Stage 1 error: {e}")
                                self._process_output()
                                raise Exception
        self._process_output()