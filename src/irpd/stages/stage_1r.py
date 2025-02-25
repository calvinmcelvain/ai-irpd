import logging
from irpd.stages.base_stage import BaseStage
from utils import file_to_string, write_file
from irpd.output_manager import StageRun

log = logging.getLogger(__name__)


class Stage1r(BaseStage):
    def __init__(self, test_config, sub_path, context, llm, max_instances, threshold):
        super().__init__(test_config, sub_path, context, llm, max_instances, threshold)
        self.stage = "1r"
        self.schema = self.schemas[self.stage]
        self.output = StageRun(self.stage)
    
    def _get_system_prompt(self):
        system_prompts = {case: {} for case in self.cases}
        for c in self.cases:
            for i in self._get_instance_types(c):
                prompt_name = f"stg_1r_{self.treatment}.md"
                prompt_path = self.prompt_path / c / self.ra / prompt_name
                system_prompts[c][i] =  file_to_string(prompt_path)
        return system_prompts
    
    def _get_user_prompt(self):
        user_prompts = {case: {} for case in self.cases}
        for c in self.cases:
            for i in self._get_instance_types(c):
                if not self.context.has("1", c, i):
                    self._update_context("1", c)
                output = self.context.get("1", c, i)
                user_prompts[c][i] = self._output_to_txt(
                    output[0], self.schemas["1"]
                )
        return user_prompts
    
    def _process_output(self):
        meta_path = self.sub_path / "_test_info" / f"stg_{self.stage}_test_info.json"
        if not meta_path.exists():
            self._write_meta()
        
        text = f"# Stage {self.stage} Output Categories\n\n"
        for c in self.cases:
            for i in self._get_instance_types(c):
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
            pdf_path = self.sub_path / f"{c}_stg_{self.stage}_categories.pdf"
            self._txt_to_pdf(text, pdf_path)
        
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
        except Exception as e:
            log.error(f"Error in running stage {self.stage}: {e}")
            