"""
Output manager module.

Contains the functional OutputManager model.
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Union, Optional, overload

from helpers.utils import check_directories, load_json_n_validate
from core.functions import requestout_to_irpdout
from core.foundation import FoundationalModel
from core.prompt_composer import PromptComposer
from core.output_writer import OutputWriter
from _types.batch_output import BatchOut
from _types.request_output import RequestOut
from _types.irpd_config import IRPDConfig
from _types.stage_output import StageOutput
from _types.test_output import TestOutput
from _types.irpd_output import IRPDOutput
from _types.irpd_meta import IRPDMeta, StageInfo, SubsetInfo


log = logging.getLogger(__name__)



class OutputManager(FoundationalModel):
    """
    OutputManager model, inherits the FoundationalModel.
    """
    def __init__(self, irpd_config: IRPDConfig):
        super.__init__(self, irpd_config)
        
        self.prompt_composer = PromptComposer(irpd_config)
        self.output_writer = OutputWriter(irpd_config)
        
        # Initializing output objects.
        self.outputs = self._initialize_outputs()
        
        # Checking current test path (and batch) for outputs on initialization.
        self._check_test_directory()
        if self.batches: self._check_batch()
        
    def _initialize_outputs(self) -> List[TestOutput]:
        """
        Initializes all TestOutput objects.
        
        Note: Outputs stored by the LLM string.
        """
        outputs = {}
        for llm_str in self.llms:
            llm_instance = self.llm_instances[llm_str]
            for n in range(1, self.total_replications + 1):
                sub_path = self._generate_subpath(n, llm_str)
                outputs[llm_str] = [TestOutput(
                    sub_path=sub_path,
                    meta_path=sub_path / self.file_names["meta"],
                    replication=n,
                    llm_str=llm_str,
                    meta=IRPDMeta(
                        model=llm_instance.model,
                        configs=llm_instance.configs
                    ),
                    stage_outputs={
                        stage: StageOutput(
                            stage_name=stage,
                            stage_path=sub_path / f"stage_{stage}",
                            outputs={subset: [] for subset in self.subsets[stage]}
                        )
                        for stage in self.stages
                    }
                )]
        return outputs
    
    def _check_test_directory(self) -> None:
        """
        Checks each possible subpath of a TestConfig for outputs.
        
        Note: This method is run before `_check_batch` method.
        """
        for test_output in self.outputs:
            for stage_output in test_output.stage_outputs.values():
                # Skip if stage directory doesn't exist
                if not stage_output.stage_path.exists(): continue
                
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
                        requestout_to_irpdout(
                            stage_name,
                            subset,
                            subset_path,
                            RequestOut(
                                parsed=load_json_n_validate(
                                    path, self.schemas[stage_name])
                            )
                        )
                        for path in responses_path.iterdir()
                    )
                self.check_output_completion(stage_output)
        
        return None
    
    def _check_batch(self) -> None:
        """
        Checks the Batch status if outputs don't exist in directory.
        """
        for llm_str, llm_instance in self.llm_instances.items():
            test_outputs = self.retrieve(llm_str)
            if all(self.check_output_completion(test) for test in test_outputs):
                continue
            
            if not test_outputs[0].meta_path.exists(): continue

            for stage_name in self.stages:
                stage_meta = test_outputs[0].meta.stages.get(stage_name)
                if not stage_meta: break

                batch_id, batch_path = stage_meta.batch_id, stage_meta.batch_path
                if not (batch_id and batch_path): break

                batch_out = llm_instance.retreive_batch(
                    batch_id, self.schemas[stage_name], Path(batch_path)
                )

                if isinstance(batch_out, BatchOut):
                    self.store_batch(llm_str, stage_name, batch_out)
                    break

                log.info(f"Batch - {batch_id}, is {batch_out}.")
        return None 
    
    def check_output_completion(
        self, output: Union[StageOutput, TestOutput]
    ) -> bool:
        """
        Checks whether all outputs for a given stage or a given test are 
        complete and sets the complete field.
        """
        if isinstance(output, StageOutput):
            stage_name = output.stage_name
            expected_outputs = self.prompt_composer.expected_outputs[stage_name]
            total_outputs = len(output.outputs.values())
            output.complete = total_outputs == expected_outputs
            return output.complete
        output.complete = all(stage.complete for stage in output.stage_outputs.values())
        return output.complete

    @overload
    def retrieve(
        self,
        llm_str: str,
        n: int,
        stage_name: str
    ) -> List[IRPDOutput]: ...

    @overload
    def retrieve(
        self,
        llm_str: str,
        n: int,
        stage_name: None = ...
    ) -> TestOutput: ...
    
    @overload
    def retrieve(
        self,
        llm_str: str,
        n: None = ...,
        stage_name: None = ...
    ) -> List[TestOutput]: ...
    
    def retrieve(
        self,
        llm_str: Optional[str] = None,
        n: Optional[int] = None,
        stage_name: Optional[str] = None
    ) -> Union[TestOutput, List[TestOutput], List[IRPDOutput]]:
        """
        Retrieves output(s). If stage_name is defined, returns a list of 
        IRPDOutput object. Otherwise returns TestOuput object(s).
        """
        outputs = self.outputs
        if llm_str:
            outputs = list(filter(lambda output: output.llm_str == llm_str, outputs))
            if n is not None:
                outputs = list(filter(lambda output: output.replication == n, outputs))[0]
                if stage_name:
                    outputs = outputs.stage_outputs[stage_name]
        return outputs
    
    def update_meta(
        self, test_output: TestOutput, stage_name: str, irpd_output: IRPDOutput
    ) -> None:
        """
        Updates test meta given an output
        """
        stage_meta = test_output.meta.stages.setdefault(stage_name, StageInfo(subsets={}))
        subset_meta = stage_meta.subsets.setdefault(irpd_output.subset, SubsetInfo())

        output_meta = irpd_output.request_out.meta

        subset_meta.created = datetime.fromtimestamp(output_meta.created)
        subset_meta.input_tokens += output_meta.input_tokens
        subset_meta.output_tokens += output_meta.output_tokens
        subset_meta.total_tokens += output_meta.total_tokens

        return None
    
    def store_output(
        self,
        llm_str: str,
        n: int,
        stage_name: str,
        irpd_output: IRPDOutput
    ) -> None:
        """
        Stores & writes a IRPDOutput object & updates meta.
        """
        test_output = self.retrieve(llm_str, n)
        subset = irpd_output.subset
        
        self.update_meta(test_output, stage_name, irpd_output)
        
        stage_output = test_output.stage_outputs[stage_name]
        stage_output.outputs[subset].append(irpd_output)
        self.check_output_completion(stage_output)
        
        self.output_writer.write_output(test_output, stage_name, subset)
        
        return None
    
    def store_batch(
        self,
        llm_str: str,
        stage_name: str,
        batch_out: BatchOut
    ):
        """
        Stores a batch request.
        """
        outputs = batch_out.responses
        test_outputs = self.retrieve(llm_str)
        
        for test_output in test_outputs:
            n  = test_output.replication
            stage_output = test_output.stage_outputs[stage_name]
            
            for subset, irpd_output in stage_output.outputs.items():
                subset_path = stage_output.stage_path / subset
                irpd_output.extend([
                    requestout_to_irpdout(
                        stage_name,
                        subset,
                        subset_path,
                        output.response
                    )
                    for output in outputs
                    if output.response_id.startswith(f"{n}-{subset}")
                ])
            self.check_output_completion(stage_output)
            
            self.output_writer.write_output(test_output, stage_name)
            
        return None
