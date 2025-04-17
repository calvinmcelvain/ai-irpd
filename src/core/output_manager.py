"""
Output manager module.

Contains the functional OutputManager model.
"""
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, fields
from typing import List, Union, Optional, overload

from helpers.utils import check_directories, load_json_n_validate
from core.functions import complete_irpdout
from core.foundation import FoundationalModel
from core.prompt_composer import PromptComposer
from core.output_writer import OutputWriter
from _types.batch_output import BatchOut
from _types.irpd_config import IRPDConfig
from _types.stage_output import StageOutput
from _types.test_output import TestOutput
from _types.irpd_output import IRPDOutput
from _types.irpd_meta import IRPDMeta, StageInfo, SubsetInfo


log = logging.getLogger("app")



class OutputManager(FoundationalModel):
    """
    OutputManager model, inherits the FoundationalModel.
    """
    def __init__(self, irpd_config: IRPDConfig):
        super().__init__(irpd_config)
        
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
        log.debug("Initializing TestOutput objects.")
        outputs = []
        for llm_str, llm_instance in self.llm_instances.items():
            for n in range(1, self.total_replications + 1):
                sub_path = self._generate_subpath(n, llm_str)
                outputs.append(TestOutput(
                    sub_path=sub_path,
                    meta_path=sub_path / self.file_names["meta"],
                    replication=n,
                    llm_str=llm_str,
                    meta=IRPDMeta(
                        model=llm_instance.model,
                        configs=llm_instance.configs,
                        test_info=asdict(self.irpd_config)
                    ),
                    stage_outputs={
                        stage: StageOutput(
                            stage_name=stage,
                            stage_path=sub_path / f"stage_{stage}",
                            outputs={subset: [] for subset in self.subsets[stage]}
                        )
                        for stage in self.stages
                    }
                ))
        log.debug("TestOutput objects initialized successfully.")
        return outputs
    
    def _check_test_directory(self) -> None:
        """
        Checks each possible subpath of a TestConfig for outputs.
        
        Note: This method is run before `_check_batch` method.
        """
        log.info("Checking test directories for existing outputs.")
        for test_output in self.outputs:
            # Validating config if test already exists.
            if test_output.meta_path.exists():
                self._validate_config(test_output.meta.test_info)
            
            for stage_output in test_output.stage_outputs.values():
                if not stage_output.stage_path.exists():
                    log.debug(
                        f"Stage path {stage_output.stage_path} does not exist."
                        " Skipping."
                    )
                    continue
                
                stage_name = stage_output.stage_name
                schema = self.schemas[stage_name]
                
                for subset, outputs in stage_output.outputs.items():
                    subset_path = stage_output.stage_path / subset
                    responses_path = subset_path / "responses"
                    prompts_path = subset_path / "prompts"
                    
                    if not check_directories([responses_path, prompts_path]):
                        log.debug(
                            f"Required directories for subset {subset} are"
                            " missing. Skipping."
                        )
                        continue
                    
                    outputs.extend([
                        IRPDOutput(parsed=load_json_n_validate(path, schema))
                        for path in responses_path.iterdir()
                    ])
                self.check_output_completion(stage_output)
        log.info("Test directory check completed.")
        return None
    
    def _check_batch(self) -> None:
        """
        Checks the Batch status if outputs don't exist in directory.
        """
        log.info("Checking batch status for outputs.")
        for llm_str, llm_instance in self.llm_instances.items():
            test_outputs = self.retrieve(llm_str)
            if all(self.check_output_completion(test) for test in test_outputs):
                log.debug(
                    f"All outputs for LLM {llm_str} are complete. Skipping"
                    " batch check."
                )
                continue
            
            if not test_outputs[0].meta_path.exists():
                log.debug(
                    f"Meta path {test_outputs[0].meta_path} does not exist."
                    " Skipping."
                )
                continue

            for stage_name in self.stages:
                stage_meta = test_outputs[0].meta.stages.get(stage_name)
                if not stage_meta:
                    log.debug(
                        f"No metadata found for stage {stage_name}. Breaking.")
                    break

                batch_id, batch_path = stage_meta.batch_id, stage_meta.batch_path
                if not (batch_id and batch_path):
                    log.debug(
                        f"Batch ID or path missing for stage {stage_name}."
                        " Breaking."
                    )
                    break

                batch_out = llm_instance.retreive_batch(
                    batch_id, self.schemas[stage_name], Path(batch_path)
                )

                if isinstance(batch_out, BatchOut):
                    self.store_batch(llm_str, stage_name, batch_out)
                    break

                log.info(f"Batch - {batch_id}, is {batch_out}.")
        log.info("Batch check completed.")
        return None
    
    def _validate_config(self, config: IRPDConfig) -> bool:
        """
        Checks if the case, ra, treatment, llms, and llm_config from the 
        IRPDConfig in meta is the same as the current IRPDConfig. This is 
        primarily for test_path validation. If a test_path is specified to 
        complete an incomplete path, the configs for that test should be the 
        same.
        """
        check_fields = {"case", "ra", "treatment", "llms", "llm_config"}
        for field in fields(config):
            if field.name not in check_fields:
                continue
            if getattr(config, field.name) != getattr(self.irpd_config, field.name):
                return False
        return True
    
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
            total_outputs = sum(
                len(subset_outputs) for subset_outputs in output.outputs.values())
            output.complete = total_outputs == expected_outputs
            log.debug(f"Stage {stage_name} completion status: {output.complete}.")
            return output.complete
        output.complete = all(
            stage.complete for stage in output.stage_outputs.values())
        log.debug(f"TestOutput completion status: {output.complete}.")
        return output.complete

    @overload
    def retrieve(
        self,
        llm_str: str,
        n: int,
        stage_name: str
    ) -> StageOutput: ...

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
    ) -> Union[TestOutput, List[TestOutput], StageOutput]:
        """
        Retrieves output(s). If stage_name is defined, returns a list of 
        IRPDOutput object. Otherwise returns TestOuput object(s).
        """
        outputs = self.outputs
        if llm_str:
            outputs = list(
                filter(lambda output: output.llm_str == llm_str, outputs))
            if n is not None:
                outputs = list(
                    filter(lambda output: output.replication == n, outputs))[0]
                if stage_name:
                    outputs = outputs.stage_outputs[stage_name]
        return outputs
    
    def update_meta(
        self,
        test_output: TestOutput,
        stage_name: str,
        subset: str,
        irpd_output: IRPDOutput
    ) -> None:
        """
        Updates test meta given an output
        """
        stage_meta = test_output.meta.stages.setdefault(
            stage_name, StageInfo(subsets={}))
        subset_meta = stage_meta.subsets.setdefault(subset, SubsetInfo())

        subset_meta.created = str(datetime.fromtimestamp(irpd_output.created))
        subset_meta.input_tokens += irpd_output.input_tokens
        subset_meta.output_tokens += irpd_output.output_tokens
        subset_meta.total_tokens += irpd_output.total_tokens

        return None
    
    def store_output(
        self,
        llm_str: str,
        n: int,
        stage_name: str,
        subset: str,
        irpd_output: IRPDOutput
    ) -> None:
        """
        Stores & writes a IRPDOutput object & updates meta.
        """
        test_output = self.retrieve(llm_str, n)
        
        self.update_meta(test_output, stage_name, subset, irpd_output)
        
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
        log.info(f"Storing batch outputs for LLM {llm_str}, stage {stage_name}.")
        outputs = batch_out.responses
        test_outputs = self.retrieve(llm_str)
        
        for test_output in test_outputs:
            n = test_output.replication
            stage_output = test_output.stage_outputs[stage_name]
            
            for subset, irpd_output_list in stage_output.outputs.items():
                subset_path = stage_output.stage_path / subset
                for output in outputs:
                    if output.response_id.startswith(f"{n}-{subset}"):
                        irpd_output = complete_irpdout(
                            stage_name,
                            subset_path,
                            output.response
                        )
                        self.update_meta(
                            test_output, stage_name, subset, irpd_output)
                        irpd_output_list.append(irpd_output)
            self.check_output_completion(stage_output)
            self.output_writer.write_output(test_output, stage_name)
        log.info(
            f"Batch outputs stored successfully for LLM {llm_str},"
            f" stage {stage_name}."
        )
        return None
