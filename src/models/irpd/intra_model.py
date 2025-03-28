import logging
from itertools import product
from typing import Optional, List, Union
from pathlib import Path

from logger import clear_logger
from utils import create_directory
from models.irpd.base_irpd import IRPDBase
from models.irpd.test_config import TestConfig
from models.irpd.test_prompts import TestPrompts
from models.irpd.stage import Stage


log = logging.getLogger(__name__)



class IntraModel(IRPDBase):
    def __init__(
        self,
        cases: Union[List[str], str],
        ras: Union[List[str], str],
        treatments: Union[List[str], str],
        stages: Union[List[str], str],
        N: int,
        llms: Optional[Union[List[str], str]] = None,
        llm_configs: Optional[Union[List[str], str]] = None,
        output_path: Optional[Union[str, Path]] = None,
        prompts_path: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        test_paths: Optional[List[str]] = None,
    ):
        super().__init__(
            cases,
            ras,
            treatments,
            stages,
            llms,
            llm_configs,
            output_path,
            prompts_path,
            data_path,
            test_paths
        )
        assert 0 < N, "`N` must be greater than 0."
        self.replications = N
        
        self.test_type = "cross_model"
        self._prod = list(product(
            self.llms, self.llm_configs, self.cases, self.ras, self.treatments
        ))
        
        self.test_paths = self._generate_test_paths()
        self._generate_configs()
    
    def _generate_test_paths(self):
        if self.test_paths:
            return self._validate_test_paths()
        test_dir = self.output_path / "cross_model"
        current_test = self._get_max_test_number(test_dir)
        test_paths = [test_dir / f"test_{i + 1 + current_test}" for i in range(len(self._prod))]
        return test_paths
    
    def _generate_configs(self):
        for idx, prod in enumerate(self._prod):
            llm, llm_config, case, ra, treatment = prod
            config = TestConfig(
                case=case,
                ra=ra,
                treatment=treatment,
                llms=llm,
                llm_config=llm_config,
                test_type=self.test_type,
                test_path=self.test_paths[idx],
                stages=self.stages
            )
            self.configs[config.id] = config
    
    def run(
        self,
        max_instances: Optional[int] = None,
        config_ids: Union[str, List[str]] = None,
        print_response: bool = False
    ):
        test = self.test_type.upper()
        test_configs = self._get_test_configs(config_ids=config_ids)
        for config in test_configs.values():
            config.max_instances = self.configs[config.id].max_instances = max_instances
            
            clear_logger(app=False)
            log.info(f"{test}: Running config = {config.id}.")
            
            create_directory(paths=config.test_path)
            
            self.output[config.id] = []
            
            llm_str = config.llms[0]
            llm = self._generate_llm_instance(
                llm=llm_str,
                config=config.llm_config,
                print_response=print_response
            )
            
            for n in self.replications:
                log.info(f"{test}: Running replication = {n}.")
                
                sub_path = config.test_path / f"replication_{n}"
                create_directory(paths=sub_path)
                
                self._update_output(
                    config=config,
                    llm=llm_str,
                    replication=n,
                    sub_path=sub_path
                )
                
                for stage_name in self.stages:
                    log.info(f"{test}: Running Stage = {stage_name}.")
                    
                    context = self._get_context(
                        config=config,
                        llm=llm_str,
                        replication=n
                    )
                    prompts = TestPrompts(
                        stage=stage_name,
                        test_config=config,
                        context=context,
                        prompt_path=self.prompts_path,
                        data_path=self.data_path
                    )
                    
                    stage_instance = Stage(
                        stage=stage_name,
                        test_config=config,
                        sub_path=sub_path,
                        llm=llm,
                        context=context,
                        prompts=prompts,
                        data_path=self.data_path
                    )
                    
                    stage_instance.run()
                    
                    idx = self._output_indx(id=config.id, llm=llm_str, replication=n)
                    self.output[config.id][idx].stage_outputs[stage_name] = stage_instance.output
                    log.info(f"{test}: Stage {stage_name} complete.")
                log.info(f"{test}: Replication {n} complete.")
            log.info(f"{test}: End of config = {config.id}")