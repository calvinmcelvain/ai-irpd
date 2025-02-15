import logging
from testing.stages.base_stage import BaseStage
from utils import file_to_string, write_file, load_json
from output_manager import StageRun
from llms.base_model import RequestOut

log = logging.getLogger("app.stage_2")


class Stage2(BaseStage):
    def __init__(self, test_config, sub_path, context, max_instances, threshold):
        super().__init__(test_config, sub_path, context, max_instances, threshold)
        self.stage = "2"
        self.schema = self.schemas[self.stage]
        self.output = StageRun(self.stage)
    
    def _update_context(self, case):
        log.info(f"OUTPUTS: Getting outputs for Stage 1c part 2, case {case}.")
        stage_run = StageRun("1c")
        path = self.sub_path / f"stage_1c" / case / "part_2" / f"stg_1c_part_2_response.txt"
        if path.exists():
            log.info("OUTPUTS: Outputs retreived.")
            response = load_json(path, True)
            stage_run.store(case, "part_2", RequestOut(response=response))
            self.context.store(stage_run)
            log.info(f"OUTPUTS: Stage 1c part 2, case {case} outputs stored in context.")
        else:
            log.warning("OUTPUTS: Stage 1c part 2 not found. Proceeding without.")
        return None
    
    def _get_system_prompt(self):
        if not self.context.has("1c", self.case, "part_2"):
            self._update_context(self.case, "1c")
        if self.context.has("1c", self.case, "part_2"):
            stg1c = self.context.get("1c", self.case, "part_2")
            stg1c_txt = self._output_to_txt(stg1c[0], self.schemas["1r"])
        else:
            stg1c_txt = ""
        
        system_prompts = {case: {} for case in self.cases}
        for c in self.cases:
            for i in self._get_instance_types(c):
                prompt_name = f"stg_1r_{self.treatment}.md"
                prompt_path = self.prompt_path / c / self.ra / prompt_name
                md_prompt = file_to_string(prompt_path)
                if not self.context.has("1r", c, i):
                    self._update_context(c, "1r")
                stg1r = self.context.get("1r", c, i)
                stg1r_txt = self._output_to_txt(stg1r[0], self.schemas["1r"])
                categories = stg1c_txt + stg1r_txt
                system_prompts[c][i] =  file_to_string(prompt_path)
        return system_prompts
    
    def _get_user_prompt(self):
        user_prompts = {case: {} for case in self.cases}
        for c in self.cases:
            for i in self._get_instance_types(c):
                if not self.context.has("1", c, i):
                    self._update_context(c, "1")
                output = self.context.get("1", c, i)
                user_prompts[c][i] = self._output_to_txt(
                    output[0], self.schemas["1"]
                )
        return user_prompts
    
    def _process_output(self):
        meta_path = self.sub_path / "_test_info" / f"stg_{self.stage}_test_info.json"
        if not meta_path.exists():
            self._write_meta()
        
        for c in self.cases:
            for i in self._get_instance_types(c):
                if not self._check_completed_requests(i, c):
                    output = self.output.get(c, i)[0]
                    
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
        self._output_to_pdf()
        
    def run(self):
        try:
            for c in self.cases:
                for i in self._get_instance_types(c):
                    if not self._check_completed_requests(i, c):
                        system_prompt = self._get_system_prompt()
                        user_prompt = self._get_user_prompt()
                        
                        output = self.llm.request(
                            user=str(user_prompt[c][i]),
                            system=str(system_prompt[c][i]),
                            schema=self.schema
                        )
                        self.output.store(c, i, output)
            
            self._process_output()
            return self.output
        except Exception as e:
            log.error(f"Error in running stage {self.stage}: {e}")
            return self.output