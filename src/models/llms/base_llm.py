"""
This module contains the BaseLLM model which acts as the base model 
for LLM client SDKs.
"""

from pydantic import BaseModel
from typing import List, Optional, Tuple
from pathlib import Path
from abc import ABC, abstractmethod

from models.prompts import Prompts
from models.request_output import RequestOut, MetaOutput
from utils import validate_json_string



class BaseLLM(ABC):
    """
    A abstract class acting as a base for LLM client SDKs.
    """
    def __init__(
        self,
        api_key: str,
        model: str,
        configs: BaseModel,
        print_response: bool = False,
        **kwargs
    ):
        """
        Initialize LLM model.

        Args:
            api_key (str): API key for respective LLM.
            model (str): LLM model.
            configs (BaseModel): LLM configs.
            print_response (bool, optional): If True, prints response of LLM. 
            Defaults to False.
        """
        self.api_key = api_key
        self.model = model
        self.configs = configs or self.default_configs()
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
    
    def _prep_messages(self, user: str, system: str):
        """
        Prepares messages for LLM.
        """
        pass
    
    def _json_tool_call(self, schema: BaseModel):
        """
        Prepares LLM load for tool call feature (used if LLM does not support
        structured outputs)
        """
        pass
    
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
        meta = MetaOutput(
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
    def default_configs(self):
        """
        Sets default configs of LLM if not specified.
        """
        pass
    
    @abstractmethod
    def create_client(self):
        """
        Initialized the LLM client.
        """
        pass
    
    def _format_batch(
        self,
        messages: List[Tuple[str, Prompts]],
        schema: BaseModel = None
    ):
        """
        Formats a list of messages to LLM batch request format.
        """
        pass
    
    def retreive_batch(
        self,
        batch_id: str,
        schema: Optional[BaseModel] = None,
        batch_file_path: Optional[Path] = None
    ):
        """
        Retrieves batch from LLM client, if complete. Otherwise returns a string
        of the current status of batch.
        """
        pass
    
    def request_batch(
        self,
        messages: List[Tuple[str, Prompts]],
        schema: Optional[BaseModel] = None,
        batch_file_path: Optional[Path] = None
    ):
        """
        Requests batch from LLM client.
        """
        pass
    
    @abstractmethod
    def request(
        self,
        prompts: Prompts,
        schema: Optional[BaseModel] = None,
        **kwargs
    ):
        """
        Requests chat completion from LLM client.
        """
        pass