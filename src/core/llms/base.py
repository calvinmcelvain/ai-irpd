"""
This module contains the BaseLLM model which acts as the base model 
for LLM client SDKs.
"""

from pydantic import BaseModel
from typing import List, Optional, Tuple
from pathlib import Path
from abc import ABC, abstractmethod

from types.llm_config import LLMConfig
from types.batch_output import BatchOut
from types.prompts import Prompts
from types.request_output import RequestOut, MetaOut
from helpers.utils import validate_json_string



class BaseLLM(ABC):
    """
    A abstract class acting as a base for LLM client SDKs.
    """
    def __init__(
        self,
        api_key: str,
        model: str,
        config: str,
        print_response: bool = False,
        **kwargs
    ):
        """
        Initialize LLM model.

        Args:
            api_key (str): API key for respective LLM.
            model (str): LLM model.
            configs (str): Name of config. Can be from ["base", "res1", "res2", 
            "res3"].
            print_response (bool, optional): If True, prints response of LLM. 
            Defaults to False.
        """
        self.api_key = api_key
        self.model = model
        self.configs = self._translate_config(config)
        self.print_response = print_response
        self.json_tool = kwargs.get("json_tool", None)
        self.batches = kwargs.get("batches", True)
        self.region = kwargs.get("region", None)
        self.base_url = kwargs.get("base_url", None)
    
    @staticmethod
    def _prep_system_message(system: str):
        """
        Prepares sytem messages.
        """
        return {"role": "system", "content": system}
    
    @staticmethod
    def _prep_user_message(user: str):
        """
        Prepares user messages.
        """
        return {"role": "user", "content": user}
    
    @staticmethod
    def _request_out(
        input_tokens: int,
        output_tokens: int,
        system: str,
        user: str,
        content: str,
        schema: str
    ):
        """
        Outputs a generalized RequestOut object from LLM response.
        """
        prompts = Prompts(system=system, user=user)
        meta = MetaOut(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        return RequestOut(
            text=content,
            parsed=validate_json_string(content, schema),
            prompts=prompts,
            meta=meta
        )
    
    @abstractmethod
    def _translate_config(self, config: str):
        """
        Translates generic config names to client specific config names.
        """
        pass
    
    @abstractmethod
    def _prep_messages(self, user: str, system: str):
        """
        Prepares messages for LLM.
        """
        pass
    
    @abstractmethod
    def _json_tool_call(self, schema: BaseModel):
        """
        Prepares LLM load for tool call feature (used if LLM does not support
        structured outputs)
        """
        pass
    
    @abstractmethod
    def _request_load(
        self,
        user: str,
        system: str,
        schema: Optional[BaseModel]
    ):
        """
        Creates the and returns general format for requests for LLM chat
        completions.
        """
    
    @abstractmethod
    def create_client(self):
        """
        Initializes the LLM client.
        """
        pass
    
    @abstractmethod
    def _format_batch(
        self,
        messages: List[Tuple[str, Prompts]],
        schema: BaseModel = None
    ):
        """
        Formats a list of messages to LLM batch request format.
        """
        pass
    
    @abstractmethod
    def retreive_batch(
        self,
        batch_id: str,
        schema: Optional[BaseModel] = None,
        batch_file_path: Optional[Path] = None
    ) -> BatchOut | str:
        """
        Retrieves batch from LLM client, if complete. Otherwise returns a 
        string of the current status of batch.

        Args:
            batch_id (str): The ID of the batch to be retrieved.
            schema (Optional[BaseModel], optional): The schema of output if 
            requested structured outputs. Defaults to None.
            batch_file_path (Optional[Path], optional): The path to the original
            batch request. Used to output a 'complete' BatchOut object. 
            Defaults to None.
        """
        pass
    
    @abstractmethod
    def request_batch(
        self,
        messages: List[Tuple[str, Prompts]],
        schema: Optional[BaseModel] = None,
        batch_file_path: Optional[Path] = None
    ) -> str:
        """
        Requests batch from LLM client. Returns the batch ID.

        Args:
            messages (List[Prompts]): A list of Prompt objects.
            schema (Optional[BaseModel], optional): The output structure/schema. 
            Defaults to None.
            batch_file_path (Optional[Path], optional): The path to save 
            formatted batch. Saves as jsonl. Defaults to None.
        """
        pass
    
    @abstractmethod
    def request(
        self,
        prompts: Prompts,
        schema: Optional[BaseModel] = None,
        **kwargs
    ) -> RequestOut:
        """
        Requests chat completion from LLM client. Returns a RequestOut
        object.

        Args:
            prompts (Prompts): A Prompt object.
            schema (BaseModel, optional): The structure/schema of output. 
            Defaults to None.
            kwargs:
                - max_attempts: Number of attempts if failure.
                - rate_limit_time: Time to wait if request limit hit.
        """
        pass