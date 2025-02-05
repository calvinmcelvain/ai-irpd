import pandas as pd
from stages.base_stage import BaseStage
from itertools import product
from stages.response_schemas import Stage1Schema
from utils import file_to_string


class Stage1(BaseStage):
    def __init__(self, test_config, outpath):
        super().__init__(test_config, outpath)
        self.schema = Stage1Schema
    
    def _get_system_prompt(self):
        system_prompts = {}
        for case, instance_type in product(self.case, self.instance_types):
            prompt_name = f"stg_1_{self.treatment}_{instance_type}.md"
            prompt_path = self.prompt_path / case / self.ra / prompt_name
            system_prompts[case][instance_type] = file_to_string(prompt_path)
        return system_prompts
    
    def _get_user_prompt(self):
        user_prompts = {}
        for case, instance_type in product(self.case, self.instance_types):
            df_name = f"{self.case}_{self.treatment}_{self.ra}_{instance_type}.csv"
            df_path = self.raw_data_path / df_name
            user_prompts[case][instance_type] = pd.read_csv(df_path).to_dict("records")
        return user_prompts