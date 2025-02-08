import logging
import pandas as pd
from stages.base_stage import BaseStage
from stages.response_schemas import Stage1Schema
from utils import file_to_string, write_file
from output_manager import StageRun

log = logging.getLogger(__name__)

class Stage1(BaseStage):
    def __init__(self, test_config, sub_path, context):
        super().__init__(test_config, sub_path, context)
        self.schema = Stage1Schema
        self.stage = "1"
        self.output = StageRun(self.stage)
    
    def _get_system_prompt(self):
        system_prompts = {case: {} for case in self.case}
        for case, instance_type in self.product_ci:
            prompt_name = f"stg_1_{self.treatment}_{instance_type}.md"
            prompt_path = self.prompt_path / case / self.ra / prompt_name
            system_prompts[case][instance_type] =  file_to_string(prompt_path)
        return system_prompts
    
    def _get_user_prompt(self):
        user_prompts = {case: {} for case in self.case}
        for case, instance_type in self.product_ci:
            df_name = f"{case}_{self.treatment}_{self.ra}_{instance_type}.csv"
            df_path = self.data_path / "test" / df_name
            user_prompts[case][instance_type] = pd.read_csv(df_path).to_dict("records")
        return user_prompts
    
    def _process_output(self):
        for case, instance_type in self.product_ci:
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
        self._write_meta()
        
    def run(self):
        try:
            for case, instance_type in self.product_ci:
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
            