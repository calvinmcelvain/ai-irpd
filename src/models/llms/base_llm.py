from pydantic import BaseModel
from typing import List
from abc import ABC, abstractmethod

from models.prompts import Prompts
from models.request_output import RequestOut
from models.meta_output import MetaOutput
from utils import validate_json_string



class BaseLLM(ABC):
    configs: BaseModel
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        configs: BaseModel = None,
        print_response: bool = False,
        json_tool: bool = False,
        **kwargs
    ):
        self.api_key = api_key
        self.model = model
        self.configs = configs or self.default_configs()
        self.print_response = print_response
        self.json_tool = json_tool
    
    @staticmethod
    def _prep_system_message(system: str):
        return {"role": "system", "content": system}
    
    @staticmethod
    def _prep_user_message(user: str):
        return {"role": "user", "content": user}
    
    def _prep_messages(self, user: str, system: str):
        pass
    
    def _json_tool_call(self, schema: BaseModel):
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
        prompts = Prompts(system=system, user=user)
        meta = MetaOutput(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt=prompts
        )
        return RequestOut(
            text=content,
            parsed=validate_json_string(content, schema),
            meta=meta
        )
    
    @abstractmethod
    def default_configs(self):
        pass
    
    @abstractmethod
    def create_client(self):
        pass
    
    @abstractmethod
    def format_batch(self, messages: List[Prompts]):
        pass
    
    @abstractmethod
    async def batch_request(self, batch_file: str):
        pass
    
    @abstractmethod
    def request(
        self,
        user: str,
        system: str,
        schema: BaseModel = None,
        **kwargs
    ):
        pass