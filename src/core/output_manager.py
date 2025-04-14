"""
Output manager module.

Contains the functional OutputManager model.
"""
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict

from helpers.utils import (
    check_directories, load_json_n_validate, to_list, lazy_import, create_directory
)
from core.functions import generate_llm_instance, instance_types
from _types.batch_output import BatchOut
from _types.request_output import RequestOut
from _types.irpd_config import IRPDConfig
from _types.stage_output import StageOutput
from _types.irpd_meta import IRPDMeta


log = logging.getLogger(__name__)



class OutputManager:
    """
    OutputManager model.
    
    Takes a given test config, and checks its respective directories, if they
    exists, and updates output for outputs that already exist. 
    
    This model also has methods `store` to store a StageOutput, or list of 
    StageOutputs, as well as `store_batch` for batch requests. 
    """
    def __init__(self, irpd_config: IRPDConfig):
        self.irpd_config = irpd_config
        self.stages = irpd_config.stages
        self.cases = irpd_config.cases
        self.batches = irpd_config.batches
        self.test_path = Path(irpd_config.test_path)
        self.llms = irpd_config.llms
        self.case = irpd_config.case
        self.ra = irpd_config.ra
        self.treatment = irpd_config.treatment
        self.llm_config = irpd_config.llm_config
        self.total_replications = irpd_config.total_replications
        self.schemas = {
            stage: lazy_import("types.irpd_stage_schemas", f"Stage{stage}Schema")
            for stage in self.stages
        }
        self.subsets = {
            stage: self._get_subsets(stage)
            for stage in self.stages
        }
        
        # Initializing StageOutput objects.
        self.irpd_outputs: Dict[str, List[StageOutput]] = self._initialize_irpd_outputs()
        
        # Checking current test path (and batch) for outputs on initialization.
        self._check_test_directory()
        if self.batches: self._check_batch()
        
    def _generate_subpath(self, n: int, llm_str: str):
        """
        Generates a subpath for a given replication and LLM. 
        
        For tests w/ more than one LLM, a dir. is made for each LLM. For tests 
        w/ more than one replication, a dir. is made for each replicaiton (w/ 
        respect to the LLM dir.)
        """
        subpath = self.test_path
        if len(self.llms) > 1: subpath = subpath / llm_str
        if self.total_replications > 1: subpath = subpath / f"replication_{n}"
        if not subpath.exists():
            create_directory(subpath)
        return subpath
    
    def _get_subsets(self, stage_name: str):
        """
        Generates subsets for a given stage.
        """
        subsets = ["full"]
        if stage_name in {"1", "1r"}:
            prod = [
                (case, instance_type)
                for case in self.cases
                for instance_type in instance_types(case)
            ]
            subsets += [f"{c}_{i}" for c, i in prod]
        return subsets
        
    def _initialize_irpd_outputs(self):
        """
        Initializes StageOutput objects.
        
        Note: IRPD outputs stored by LLM.
        """
        irpd_outputs = {}
        for llm_str in self.irpd_config.llms:
            irpd_outputs[llm_str] = []
            for n in range(1, self.total_replications + 1):
                sub_path = self._generate_subpath(n, llm_str)
                for stage in self.stages:
                    for subset in self.subsets[stage]:
                        output_path = sub_path / f"stage_{stage}" / subset
                        # Creates a StageOutput object for each replication,
                        # stage, and subset.
                        irpd_outputs[llm_str].append(StageOutput(
                            stage_name=stage,
                            subset=subset,
                            llm_str=llm_str,
                            replication=n,
                            output_path=output_path
                        ))
        return irpd_outputs
    
    def _generate_meta_path(self, n: int, llm_str: str):
        """
        Generates the path for 'test' meta. File exists for each subpath.
        """
        subpath = self._generate_subpath(n, llm_str)
        return subpath / "_test_meta.json"
    
    def _check_test_directory(self):
        """
        Checks each possible subpath of a TestConfig for outputs.
        
        Note: This method is run before `_check_batch` method.
        """
        for llm_str, test_output in self.irpd_outputs.items():
            for stage_output in test_output:
                responses_path = stage_output.output_path / "responses"
                prompts_path = stage_output.output_path / "prompts"
                
                # If prompts & responses directories don't exist, there are no
                # outputs to store.
                if not check_directories([responses_path, prompts_path]):
                    continue
                
                stage_name = stage_output.stage_name
                stage_output.outputs = [
                    RequestOut(
                        parsed=load_json_n_validate(path, self.schemas[stage_name])
                    )
                    for path in responses_path.iterdir()
                ]
                
                self.store_completion(stage_output)
        return None
    
    def _check_batch(self):
        """
        Checks the Batch status if outputs don't exist in directory.
        """
        for llm_str in self.llms:
            # Check to see if the outputs are complete already.
            if self.check_irpd_test_completion(llm_str): continue
            
            # Meta paths are different across replications, but all replications 
            # are done in one batch, thus the batch ID in meta is the same.
            meta_path = self._generate_meta_path(llm_str, 1)
            if not meta_path.exists(): continue
                
            meta: IRPDMeta = load_json_n_validate(meta_path, IRPDMeta)
            
            # Initializing LLM to check batches.
            llm = generate_llm_instance(llm_str, self.llm_config)
            for stage_name in self.stages:
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
                
                # Checking batch status.
                batch_out = llm.retreive_batch(
                    batch_id, self.schemas[stage_name], Path(batch_path)
                )
                
                # If batch complete, it is a BatchOut object.
                if isinstance(batch_out, BatchOut):
                    self.store_batch(
                        llm_str, stage_name, batch_out, batch_path
                    )
                    break
                
                # If batch incomplete, skipped for now. See TestRunner
                # for retry loop in waiting for batch.
                log.info(f"Batch - {batch_id}, is {batch_out}.")
        return None
    
    def _get_output_index(self, stage_output: StageOutput):
        """
        Returns the index of a given StageOutput object within irpd_outputs.
        """
        return self.irpd_outputs[stage_output.llm_str].index(stage_output)
    
    def _log_stored_completion(self, stage_output: StageOutput):
        """
        Logs stored completion.
        """
        log.info(
            "\nOutputs stored successfully for:"
            f"\n\t config: {self.irpd_config.id}"
            f"\n\t case: {self.irpd_config.case}"
            f"\n\t llm: {stage_output.llm_str}"
            f"\n\t replicate: {stage_output.replication} of {self.total_replications}"
            f"\n\t stage: {stage_output.stage_name}"
            f"\n\t subset: {stage_output.subset}"
            f"\n\t COMPLETE: {stage_output.complete}"
        )
        return None
    
    def check_stage_completion(self, llm_str: str, stage_name: str, n: int):
        """
        Checks whether a stage is complete.
        """
        all_stage_ouptuts = self.retrieve(llm_str, n, stage_name)
        return all(output.complete for output in all_stage_ouptuts)
    
    def check_irpd_test_completion(self, llms: Union[str, List[str]]):
        """
        Checks if all outputs for an LLM(s) are complete. Returns a boolean.
        """
        llm_strs = to_list(llms)
        for llm_str in llm_strs:
            llm_output: List[StageOutput] = self.irpd_outputs[llm_str]
            complete = all(stage_output.complete for stage_output in llm_output)
            if not complete: return False
        return True
    
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
        outputs = self.irpd_outputs
        if llm_str:
            outputs = outputs[llm_str]
            if n is not None:
                outputs = list(filter(lambda output: output.replication == n, outputs))
            if stage_name:
                outputs = list(filter(lambda output: output.stage_name == stage_name, outputs))
            if subset:
                outputs = list(filter(lambda output: output.subset == subset, outputs))
        return outputs
    
    def store_completion(self, stage_output: StageOutput):
        """
        Stores a chat completion request.
        """
        llm_str = stage_output.llm_str
        n = stage_output.replication
        stage_name = stage_output.stage_name
        subset = stage_output.subset
        
        output: StageOutput = self.retrieve(llm_str, n, stage_name, subset)
        
        output.outputs = to_list(stage_output.outputs)
        output.complete = True
        
        idx = self._get_output_index(output)
        self.irpd_outputs[stage_output.llm_str][idx] = output
        
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
            
            idx = self._get_output_index(stage_output)
            self.irpd_outputs[llm_str][idx] = stage_output
            
            self._log_stored_completion(stage_output)
        return None
