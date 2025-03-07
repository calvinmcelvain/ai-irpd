from pydantic import BaseModel
from math import fsum
from datetime import datetime
from openai.types.chat import ChatCompletion
from openai.types import CompletionUsage
from abc import ABC, abstractmethod


class RequestOut(BaseModel):
    response: str | object
    system: str = None
    user: str = None
    meta: ChatCompletion = None


class Base(ABC):
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
    
    def _process_output(
        self,
        id: int,
        tokens: dict,
        content: str,
        system: str,
        user: str
    ) -> RequestOut:
        usage = CompletionUsage(
            completion_tokens=tokens["output_tokens"],
            prompt_tokens=tokens["input_tokens"],
            total_tokens=fsum(tokens.values())
        )
        created = int(datetime.now().timestamp())
        meta_data = ChatCompletion(
            id=id, created=created, model=self.model, object='chat.completion',
            choices=[], usage=usage
        )
        return RequestOut(
            response=content, meta=meta_data, system=system, user=user
        )
    
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
    
    @abstractmethod
    def default_configs(self):
        pass
    
    @abstractmethod
    def create_client(self):
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