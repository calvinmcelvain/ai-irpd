import logging
from pathlib import Path
from time import sleep
from typing import List, Dict, Optional, Union

from utils import check_directories, load_json_n_validate, lazy_import, to_list
from models.batch_output import BatchOut
from models.request_output import RequestOut
from models.irpd.output_processer import OutputProcesser
from models.irpd.test_config import TestConfig
from models.irpd.config_manager import ConfigManager
from models.irpd.test_outputs import TestOutput, StageOutput, TestMeta


log = logging.getLogger(__name__)



class OutputManager:
    def __init__(self, test_config: TestConfig):
        self.config_manager = ConfigManager(test_config)
        self.test_config = test_config
        self.processor = OutputProcesser
        self.test_path = test_config.test_path
        self.total_replications = test_config.total_replications
        self.stages = test_config.stages
        self.llms = test_config.llms
        self.llm_config = test_config.llm_config
        self.schemas = {
            stage: lazy_import("models.irpd.schemas", f"Stage{stage}Schema")
            for stage in self.stages
        }
        self.generate_llm_instance = self.config_manager.generate_llm_instance
        
        self.test_outputs = self._initialize_test_outputs()
        
        self._check_test_directory()
        self._check_batch()
        self._check_completeness()
        
    def _initialize_test_outputs(self):
        test_outputs = {}
        for llm_str in self.test_config.llms:
            test_output = TestOutput(llm_str=llm_str,)
            for n in range(1, self.total_replications + 1):
                for stage in self.stages:
                    subsets = self.config_manager.get_subsets(stage)
                    for subset in subsets:
                        test_output.stage_outputs.append(StageOutput(
                            stage_name=stage,
                            subset=subset,
                            llm_str=llm_str,
                            replication=n
                        ))
            test_outputs[llm_str] = test_output
        return test_outputs
    
    def _check_test_directory(self):
        for llm_str in self.llms:
            for stage_output in self.test_outputs[llm_str].stage_outputs:
                stage_name = stage_output.stage_name
                subset = stage_output.subset
                sub_path = self.config_manager.generate_subpath(stage_output.replication, stage_output.llm_str)
                stage_string = f"stage_{stage_name}"
                responses_path = sub_path / stage_string / subset / "responses"
                prompts_path = sub_path / stage_string / subset / "prompts"
                
                if not check_directories([responses_path, prompts_path]):
                    continue
                
                outputs = [
                    RequestOut(parsed=load_json_n_validate(path, self.schemas[stage_name]))
                    for path in responses_path.iterdir()
                ]
                
                stage_output.complete = True
                self.store_completion(stage_output, outputs)
        return None
    
    def _check_batch(self):
        for llm_str in self.llms:
            llm = self.generate_llm_instance(llm_str)
            test_output = self.test_outputs[llm_str]
            if not test_output.complete:
                for stage_name in self.test_config.stages:
                    stage_outputs = self.retrieve(llm_str=llm_str, stage_name=stage_name)
                    if not self._check_output_set_completeness(stage_outputs):
                        meta_path = self.config_manager.generate_meta_path(llm_str, 1)
                        if self.test_config.batches and meta_path.exists():
                            meta: TestMeta = load_json_n_validate(meta_path, TestMeta)
                            
                            if not stage_name in meta.stages.keys():
                                break
                            
                            batch_id = meta.stages[stage_name].batch_id
                            batch_path = Path(meta.stages[stage_name].batch_path)
                            
                            if batch_id is None:
                                continue
                            
                            retries = 0
                            while retries < 6:
                                schema = self.schemas[stage_name]
                                batch_out = llm.retreive_batch(batch_id, schema, batch_path)
                                
                                if isinstance(batch_out, BatchOut):
                                    self.store_batch(
                                        llm_str, stage_name, batch_out, batch_path
                                    )
                                    break
                                
                                if retries < 6:
                                    time_to_wait = 10 + retries * 10
                                    log.info(f"Waiting {time_to_wait} seconds.")
                                    sleep(time_to_wait)
                                    retries += 1
                                else:
                                    log.warning(f"Retries exhausted.")
                                    break
        return None
    
    def _get_output_index(self, stage_output: StageOutput):
        return self.test_outputs[stage_output.llm_str].stage_outputs.index(stage_output)
    
    def _check_completeness(self, llm_str: str = None):
        llm_str = to_list(llm_str) if llm_str else self.llms
        for llm in llm_str:
            self.test_outputs[llm].test_complete()
            if self.test_outputs[llm].complete:
                log.info(f"All outputs for {llm} are complete.")
        return None
    
    def _check_output_set_completeness(self, stage_outputs: List[StageOutput]):
        return all(output.complete for output in stage_outputs)
    
    def _check_stage_completion(self, stage_outputs: List[StageOutput]):
        llm_str = stage_outputs[0].llm_str
        stage_name = stage_outputs[0].stage_name
        n = stage_outputs[0].replication
        all_stage_ouptuts = self.retrieve(llm_str, n, stage_name)
        return self._check_output_set_completeness(all_stage_ouptuts)
    
    def _log_stored_completion(self, stage_output: StageOutput):
        log.info(
            "\nOutputs stored successfully for:"
            f"\n\t config: {self.test_config.id}"
            f"\n\t llm: {stage_output.llm_str}"
            f"\n\t replicate: {stage_output.replication} of {self.total_replications}"
            f"\n\t stage: {stage_output.stage_name}"
            f"\n\t subset: {stage_output.subset}"
            f"\n\t COMPLETE: {stage_output.complete}"
        )
        return None
    
    def retrieve(
        self,
        llm_str: Optional[str] = None,
        n: Optional[int] = None,
        stage_name: Optional[str] = None,
        subset: Optional[str] = None
    ):
        outputs: Dict[str, TestOutput] = self.test_outputs
        if llm_str:
            outputs: List[StageOutput] = outputs[llm_str].stage_outputs
            if n is not None:
                outputs: List[StageOutput] = list(filter(lambda output: output.replication == n, outputs))
            if stage_name:
                outputs: List[StageOutput] = list(filter(lambda output: output.stage_name == stage_name, outputs))
            if subset:
                outputs: StageOutput = list(filter(lambda output: output.subset == subset, outputs))
        if outputs is None:
            log.warning(
                "\nOutputs not found for:"
                f"\n\t config: {self.test_config.id}"
                f"\n\t llm: {llm_str}"
                f"\n\t replicate: {n} of {self.test_config.total_replications}"
                f"\n\t stage: {stage_name}"
                f"\n\t subset: {subset}"
            )
            return None
        return to_list(outputs)
    
    def store_completion(
        self,
        stage_output: StageOutput,
        outputs: Union[RequestOut, List[RequestOut]]
    ):
        llm_str = stage_output.llm_str
        n = stage_output.replication
        stage_name = stage_output.stage_name
        subset = stage_output.subset
        
        output = self.retrieve(llm_str, n, stage_name, subset)[0]
        assert isinstance(output, StageOutput), "Output could not be stored."
        
        output.outputs = to_list(outputs)
        
        idx = self._get_output_index(stage_output)
        self.test_outputs[stage_output.llm_str].stage_outputs[idx] = output
        
        self._log_stored_completion(output)
        return None
    
    def store_batch(
        self,
        llm_str: str,
        stage_name: str,
        batch_out: BatchOut,
        batch_file_path: Path
    ):
        outputs = batch_out.responses
        stage_outputs: List[StageOutput] = self.retrieve(
            llm_str=llm_str,
            stage_name=stage_name
        )
        for stage_output in stage_outputs:
            assert isinstance(stage_output, StageOutput), "Output could not be stored."
            
            n  = stage_output.replication
            subset = stage_output.subset
            
            stage_output.outputs = [
                response.response for response in outputs
                if response.response_id.startswith(f"{n}-{subset}")
            ]
            
            if not stage_output.batch_id or not stage_output.batch_path:
                stage_output.batch_id = batch_out.batch_id
                stage_output.batch_path = batch_file_path
            
            idx = self._get_output_index(stage_output)
            
            self.write_output(to_list(stage_output))
            stage_output.complete = True
            
            self.test_outputs[llm_str].stage_outputs[idx] = stage_output

            self._log_stored_completion(stage_output)
            
            replication_outputs = self.retrieve(llm_str, n, stage_name)
            self.write_output(to_list(replication_outputs))
        return None
    
    def write_output(self, stage_outputs: List[StageOutput]):
        self.processor(stage_outputs, self.config_manager).process()
        if self._check_stage_completion(stage_outputs):
            llm_str = stage_outputs[0].llm_str
            stage_name = stage_outputs[0].stage_name
            n = stage_outputs[0].replication
            all_stage_ouptuts = self.retrieve(llm_str, n, stage_name)
            self.processor(to_list(all_stage_ouptuts), self.config_manager).process(True)
            self._check_completeness(llm_str)
        return None
