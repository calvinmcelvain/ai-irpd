import pandas as pd
import logging
from stages.base_stage import BaseStage
from utils import file_to_string, write_file, validate_json_string, txt_to_pdf
from output_manager import StageRun

log = logging.getLogger(__name__)


class Stage1c(BaseStage):
    def __init__(self, test_config, sub_path, context):
        super().__init__(test_config, sub_path, context)
        self.stage = "1c"
        self.output = StageRun(self.stage)
        self.instance_types = ["part_1", "part_2"]
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
    
    def _output_to_pdf(self):
        for c in self.cases:
            text = f"# {c.upper()} Stage {self.stage} Categories\n\n"
            
            for i in self.instance_types:
                output = self.output.get(c, i)[0]
                json_output = validate_json_string(
                    output.response, self.schemas[self.schema_map[i]]
                )
                
                if json_output:
                    categories = self._get_category_att(json_output)
                    text += self._format_categories(
                        categories,
                        f"## {i.capitalize()} Categories\n\n"
                    )
            path = self.sub_path / f"{c}_stg_{self.stage}_categories.pdf"
            txt_to_pdf(text, path)
        return None
    
    def _process_output(self):
        meta_path = self.sub_path / "_test_info" / f"stg_{self.stage}_test_info.json"
        if not meta_path.exists():
            self._write_meta()
        
        for part in self.instance_types:
            output = self.output.get(self.case, part)[0]
            
            write_path = self.sub_path / f"stage_{self.stage}" / self.case / part
            write_path.mkdir(exist_ok=True, parents=True)
            prefix = f"stg_{self.stage}_{part}_"
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
            for part in self.instance_types:
                if not self._check_completed_requests(part, self.case):
                    system_prompt = self._get_system_prompt()
                    user_prompt = self._get_user_prompt()
                    
                    output = self.llm.request(
                        user=str(user_prompt),
                        system=str(system_prompt),
                        schema=self.schemas[self.schema_map[part]]
                    )
                    self.output.store(self.case, part, output)
            
            self._process_output()
            return self.output
        except Exception as e:
            log.error(f"Error in running stage {self.stage}: {e}")
            return self.output
            