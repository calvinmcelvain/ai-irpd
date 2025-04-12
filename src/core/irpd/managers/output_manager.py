"""
Output manager module.

Contains the functional OutputManager model.
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union

from helpers.utils import check_directories, load_json_n_validate, lazy_import, to_list
from types.batch_output import BatchOut
from types.request_output import RequestOut
from types.irpd_config import TestConfig
from types.irpd_output import TestOutput, StageOutput, TestMeta
from core.irpd.processors.output_processer import OutputProcesser
from core.irpd.managers.config_manager import ConfigManager


log = logging.getLogger(__name__)



class OutputManager:
    """
    OutputManager model.
    
    Takes a given test config, and checks its respective directories, if they
    exists, and updates output for outputs that already exist. 
    
    This model also has methods `store` to store a StageOutput, or list of 
    StageOutputs, as well as `store_batch` for batch requests. 
    
    Also contains method `write_output` that creates instance of the 
    OutputProcessor model to write a subset of outputs.
    """
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
            stage: lazy_import("types.irpd_stage_schemas", f"Stage{stage}Schema")
            for stage in self.stages
        }
        self.generate_llm_instance = self.config_manager.generate_llm_instance
        
        self.test_outputs = self._initialize_test_outputs()
        
        # Checking current test path (and batch) for outputs on initialization.
        self._check_test_directory()
        self._check_batch()
        
    def _initialize_test_outputs(self):
        """
        Initializes TestOutput and StageOutput objects.
        
        Note: The number of StageOutput objects within a TestOutput should be 
        the same across TestOutputs.
        """
        test_outputs = {}
        for llm_str in self.test_config.llms:
            # Creating a TestOuput object for each LLM.
            test_output = TestOutput(llm_str=llm_str,)
            for n in range(1, self.total_replications + 1):
                for stage in self.stages:
                    subsets = self.config_manager.get_subsets(stage)
                    for subset in subsets:
                        # Creates a StageOutput object for each replication,
                        # stage, and subset.
                        test_output.stage_outputs.append(StageOutput(
                            stage_name=stage,
                            subset=subset,
                            llm_str=llm_str,
                            replication=n
                        ))
            test_outputs[llm_str] = test_output
        return test_outputs
    
    def _check_test_directory(self):
        """
        Checks each possible subpath of a TestConfig for outputs.
        
        Note: This method is run before `_check_batch` method.
        """
        for llm_str, test_output in self.test_outputs.items():
            test_output: TestOutput
            for stage_output in test_output.stage_outputs:
                stage_name = stage_output.stage_name
                subset = stage_output.subset
                
                sub_path = self.config_manager.generate_subpath(
                    stage_output.replication, stage_output.llm_str
                )
                stage_string = f"stage_{stage_name}"
                responses_path = sub_path / stage_string / subset / "responses"
                prompts_path = sub_path / stage_string / subset / "prompts"
                
                # If prompts & responses directories don't exist, there are no
                # outputs to store.
                if not check_directories([responses_path, prompts_path]):
                    continue
                
                outputs = [
                    RequestOut(parsed=load_json_n_validate(path, self.schemas[stage_name]))
                    for path in responses_path.iterdir()
                ]
                
                self.store_completion(stage_output, outputs)
                
                # Checking if TestOutput object is complete.
                test_output.check_test_complete()
        return None
    
    def _check_batch(self):
        """
        Checks the Batch status if outputs don't exist in directory.
        """
        for llm_str, test_output in self.test_outputs.items():
            test_output: TestOutput
            
            # Check to see if the TestOutput object was already marked complete.
            # Note: If output found in directory, object should be marked as 
            # complete.
            if not test_output.complete:
                meta_path = self.config_manager.generate_meta_path(llm_str, 1)
                
                # Checking to see if meta path exists & Test include batches.
                if self.test_config.batches and meta_path.exists():
                    meta: TestMeta = load_json_n_validate(meta_path, TestMeta)
                    
                    # Initializing LLM to check batches (if exist).
                    llm = self.generate_llm_instance(llm_str)
                    for stage_name in self.test_config.stages:
                        # If stage name is not in meta, then breaks loop (since
                        # stages are sequential).
                        if not stage_name in meta.stages.keys():
                            break
                        
                        batch_id = meta.stages[stage_name].batch_id
                        batch_path = meta.stages[stage_name].batch_path
                        
                        if not (batch_id and batch_path):
                            # Means the batch hasn't been requested yet. Thus
                            # subsequent stages haven't.
                            break
                        else:
                            # Storing the batch ID and path in the StageOutput 
                            # objects if they exist.
                            for stage_output in test_output.stage_outputs:
                                stage_output.batch_id = batch_id
                                stage_output.batch_path = Path(batch_path)
                        
                        # Checking batch status.
                        batch_path = Path(batch_path)
                        schema = self.schemas[stage_name]
                        batch_out = llm.retreive_batch(
                            batch_id, schema, batch_path
                        )
                        
                        # If batch complete, it is a BatchOut object.
                        if isinstance(batch_out, BatchOut):
                            self.store_batch(
                                llm_str, stage_name, batch_out, batch_path
                            )
                            break
                        
                        # If batch incomplete, skipped for now. See TestRunner
                        # for retry loop in waiting for batch.
                        log.info(f"Batch - {batch_id}, was skipped from being stored.")
                    
                    # Checking to see if TestOutput object is complete.
                    test_output.check_test_complete()
                    
                    # Re-storing the TestOutput object.
                    self.test_outputs[llm_str] = test_output
        return None
    
    def _get_output_index(self, stage_output: StageOutput):
        """
        Returns the index of a given StageOutput object within TestOutput object.
        """
        return self.test_outputs[stage_output.llm_str].stage_outputs.index(stage_output)
    
    def _check_stage_completion(self, llm_str: str, stage_name: str, n: int):
        """
        Checks whether a stage is complete.
        """
        all_stage_ouptuts = self.retrieve(llm_str, n, stage_name)
        return all(output.complete for output in all_stage_ouptuts)
    
    def _log_stored_completion(self, stage_output: StageOutput):
        """
        Logs stored completion.
        """
        log.info(
            "\nOutputs stored successfully for:"
            f"\n\t config: {self.test_config.id}"
            f"\n\t case: {self.test_config.case}"
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
        """
        Retrieves output(s).
        """
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
                f"\n\t case: {self.test_config.case}"
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
        """
        Stores a chat completion request.
        """
        llm_str = stage_output.llm_str
        n = stage_output.replication
        stage_name = stage_output.stage_name
        subset = stage_output.subset
        
        output: TestOutput = self.retrieve(llm_str, n, stage_name, subset)[0]
        idx = self._get_output_index(output)
        
        output.outputs = to_list(outputs)
        output.complete = True
        self.test_outputs[stage_output.llm_str].stage_outputs[idx] = output
        
        # Writing output
        self.write_output(output)
        
        self._log_stored_completion(output)
        return None
    
    def store_batch(
        self,
        llm_str: str,
        stage_name: str,
        batch_out: BatchOut,
        batch_file_path: Path
    ):
        """
        Stores a batch request.
        """
        outputs = batch_out.responses
        stage_outputs: List[StageOutput] = self.retrieve(
            llm_str=llm_str,
            stage_name=stage_name
        )
        for stage_output in stage_outputs:
            idx = self._get_output_index(stage_output)
            n  = stage_output.replication
            subset = stage_output.subset
            
            # Storing batch outputs in StageOutput object if batch request ID
            # matches the StageOutput attrbs.
            stage_output.outputs = [
                response.response for response in outputs
                if response.response_id.startswith(f"{n}-{subset}")
            ]
            stage_output.complete = True
            
            # Storing batch id and batch path if one or both are not stored.
            if not (stage_output.batch_id or stage_output.batch_path):
                stage_output.batch_id = batch_out.batch_id
                stage_output.batch_path = batch_file_path
            
            self.test_outputs[llm_str].stage_outputs[idx] = stage_output
            
            # Writing output
            self.write_output(stage_output)

            self._log_stored_completion(stage_output)
        return None
    
    def write_output(self, stage_output: StageOutput):
        """
        Writes output
        """
        self.processor(to_list(stage_output), self.config_manager).process()
        
        llm_str = stage_output.llm_str
        stage_name = stage_output.stage_name
        n = stage_output.replication
        
        # Checking to see if stage is complete. If so, writing the final form
        # outputs (e.g., category pdfs & classification CSVs).
        stage_complete = self._check_stage_completion(llm_str, stage_name, n)
        if stage_complete:
            all_stage_ouptuts = self.retrieve(llm_str, n, stage_name)
            self.processor(to_list(all_stage_ouptuts), self.config_manager).process(True)
        return None
