"""
Output manager module.

Contains the functional OutputManager model.
"""
import logging
from pathlib import Path
from typing import List, Optional

from helpers.utils import check_directories, load_json_n_validate, to_list
from core.foundation import FoundationalModel
from core.prompt_composer import PromptComposer
from _types.batch_output import BatchOut
from _types.request_output import RequestOut
from _types.irpd_config import IRPDConfig
from _types.stage_output import StageOutput
from _types.test_output import TestOutput
from _types.irpd_output import IRPDOutput
from _types.irpd_meta import IRPDMeta


log = logging.getLogger(__name__)



class OutputManager(FoundationalModel):
    """
    OutputManager model, inherits the FoundationalModel.
    """
    def __init__(self, irpd_config: IRPDConfig):
        super.__init__(self, irpd_config)
        
        self.prompt_composer = PromptComposer(irpd_config)
        
        # Initializing output objects.
        self.outputs: List[TestOutput] = self._initialize_outputs()
        
        # Checking current test path (and batch) for outputs on initialization.
        self._check_test_directory()
        if self.batches: self._check_batch()
        
    def _initialize_outputs(self):
        """
        Initializes all TestOutput objects.
        
        Note: Outputs stored by the LLM string.
        """
        outputs = {}
        for llm_str in self.llms:
            outputs[llm_str] = [
                TestOutput(
                    sub_path=self._generate_subpath(n, llm_str),
                    meta_path=self._generate_meta_path(n, llm_str),
                    replication=n,
                    llm_str=llm_str,
                    meta=IRPDMeta(),
                    stage_outputs={
                        stage: StageOutput(
                            stage_name=stage,
                            stage_path=self._generate_subpath(n, llm_str) / f"stage_{stage}",
                            outputs={subset: [] for subset in self.subsets[stage]}
                        )
                        for stage in self.stages
                    }
                )
                for n in range(1, self.total_replications + 1)
            ]
        return outputs
    
    def _check_test_directory(self):
        """
        Checks each possible subpath of a TestConfig for outputs.
        
        Note: This method is run before `_check_batch` method.
        """
        for test_output in self.outputs:
            for stage_output in test_output.stage_outputs.values():
                # Skip if stage directory doesn't exist
                if not stage_output.stage_path.exists():
                    continue
                
                stage_name = stage_output.stage_name
                for subset, outputs in stage_output.outputs.items():
                    subset_path = stage_output.stage_path / subset
                    responses_path = subset_path / "responses"
                    prompts_path = subset_path / "prompts"
                    
                    # Skip if prompts & responses directories don't exist
                    if not check_directories([responses_path, prompts_path]):
                        continue
                    
                    # Load and extend outputs for the subset
                    outputs.extend(
                        IRPDOutput(
                            request_out=RequestOut(
                                parsed=load_json_n_validate(
                                    path, self.schemas[stage_name]
                                )
                            ),
                            subset=subset,
                            response_path=path,
                            user_path=None,
                            system_path=None
                        )
                        for path in responses_path.iterdir()
                    )
        return None
    
    def _check_batch(self):
        """
        Checks the Batch status if outputs don't exist in directory.
        """
        for llm_str in self.llms:
            # Check to see if the outputs are complete already.
            if self.check_output_completion(llm_str): continue
            
            # Meta paths are different across replications, but all replications 
            # are done in one batch, thus the batch ID in meta is the same.
            meta_path = self._generate_meta_path(llm_str, 1)
            if not meta_path.exists(): continue
                
            meta: IRPDMeta = load_json_n_validate(meta_path, IRPDMeta)
            
            # Initializing LLM to check batches.
            llm = self._generate_llm_instance(llm_str)
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
    
    def check_output_completion(
        self,
        llm_str: str,
        stage_name: str,
        n: Optional[int] = None
    ):
        """
        Checks whether all outputs for given parameters are complete.
        """
        outputs = self.retrieve(llm_str, n)
        total_outputs = len(self.retrieve(llm_str, n, stage_name))
        expected_outputs = self.prompt_composer.expected_outputs(outputs, stage_name)
        return total_outputs >= expected_outputs
    
    def retrieve(
        self,
        llm_str: Optional[str] = None,
        n: Optional[int] = None,
        stage_name: Optional[str] = None,
        subset: Optional[str] = None
    ):
        """
        Retrieves output(s). If stage_name and/or subset is defined, returns a 
        list of IRPDOutput object. Otherwise returns list of TestOuput objects.
        """
        outputs = self.outputs
        if llm_str:
            outputs = list(filter(lambda output: output.llm_str == llm_str, outputs))
        if n is not None:
            outputs = list(filter(lambda output: output.replication == n, outputs))
        if stage_name:
            outputs: List[IRPDOutput] = list(filter(
                lambda output: output.stage_outputs[stage_name].outputs, outputs))
            if subset:
                outputs: List[IRPDOutput] = list(filter(
                    lambda output: output.subset == subset, outputs))
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
