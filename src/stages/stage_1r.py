import logging
from stages.base_stage import BaseStage
from utils import file_to_string, write_file
from output_manager import StageRun

log = logging.getLogger(__name__)


class Stage1r(BaseStage):
    def __init__(self, test_config, sub_path, context):
        super().__init__(test_config, sub_path, context)
        self.stage = "1r"
        self.schema = self.schemas[self.stage]
        self.output = StageRun(self.stage)
    
    def _get_system_prompt(self):
        system_prompts = {case: {} for case in self.cases}
        for case, instance_type in self.product_ci:
            prompt_name = f"stg_1r_{self.treatment}.md"
            prompt_path = self.prompt_path / case / self.ra / prompt_name
            system_prompts[case][instance_type] =  file_to_string(prompt_path)
        return system_prompts
    
    def _get_user_prompt(self):
        user_prompts = {case: {} for case in self.cases}
        for case, instance_type in self.product_ci:
            if not self.context.has("1", case, instance_type):
                self._update_context(case, "1")
            output = self.context.get("1", case, instance_type)
            user_prompts[case][instance_type] = self._output_to_txt(
                output[0], self.schemas["1"], instance_type
            )
        return user_prompts
    
    def _process_output(self):
        for case, instance_type in self.product_ci:
            if not self._check_completed_requests(instance_type, case):
                output = self.output.get(case, instance_type)[0]
                
                write_path = self.sub_path / f"stage_{self.stage}" / case / instance_type
                write_path.mkdir(exist_ok=True, parents=True)
                prefix = f"stg_{self.stage}_{instance_type}_"
                system_path = write_path / (prefix + "sys_prmpt.txt")
                user_path = write_path / (prefix + "user_prmpt.txt")
                response_path = write_path / (prefix + "response.txt")
                
                write_file(system_path, output.system)
                write_file(user_path, output.user)
                write_file(response_path, output.response)
        self._output_to_pdf()
        
        meta_path = self.sub_path / "_test_info" / f"stg_{self.stage}_test_info.json"
        if not meta_path.exists():
            self._write_meta()
        
    def run(self):
        try:
            for case, instance_type in self.product_ci:
                if not self._check_completed_requests(instance_type, case):
                    system_prompt = self._get_system_prompt()
                    user_prompt = self._get_user_prompt()
                    
                    output = self.llm.request(
                        user=str(user_prompt[case][instance_type]),
                        system=str(system_prompt[case][instance_type]),
                        schema=self.schema
                    )
                    self.output.store(case, instance_type, output)
            
            self._process_output()
            return self.output
        except Exception as e:
            log.error(f"Error in running stage {self.stage}: {e}")
            return self.output
            