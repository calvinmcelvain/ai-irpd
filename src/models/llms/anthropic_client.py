import logging
import time
import random as r
from typing import List, Optional
from anthropic import Anthropic
from anthropic import InternalServerError, BadRequestError, RateLimitError
from anthropic.types.message import Message
from pydantic import BaseModel
from abc import abstractmethod

from models.prompts import Prompts
from models.llms.base_llm import BaseLLM


log = logging.getLogger(__name__)



class AnthropicToolCall(BaseModel):
    name: str = "json_output"
    input_schema: object | None
    

class AnthropicClient(BaseLLM):
    @abstractmethod
    def default_configs(self):
        pass
    
    def create_client(self):
        return Anthropic(api_key=self.api_key)
    
    def _json_tool_call(self, schema: BaseModel):
        tool_load = AnthropicToolCall(input_schema=schema.model_json_schema())
        tool_choice = {"name": "json_output", "type": "tool"}
        return {"tools": [tool_load.model_dump()], "tool_choice": tool_choice}
    
    def _prep_messages(self, user: str, system: str):
        messages = {"messages": [self._prep_user_message(user)]}
        messages.update({"system": system})
        return messages
    
    def _format_batch(
        self,
        messages: List[Prompts],
        message_ids: List[str],
        schema: Optional[BaseModel] = None
    ):
        pass
    
    def batch_status(self, batch_id: str):
        pass
    
    def retreive_batch(self, batch_id: str, schema: Optional[BaseModel]):
        pass
    
    def request_batch(
        self,
        messages: List[Prompts],
        message_ids: List[str],
        schema: Optional[BaseModel] = None
    ):
        pass
        
    def request(self, user: str, system: str, schema: BaseModel = None, **kwargs):
        client = self.create_client()
        
        request_load = {"model": self.model}
        request_load.update(self.configs.model_dump(exclude_none=True))
        request_load.update(self._prep_messages(user, system))
        request_load.update(self._json_tool_call(schema)) if schema else {}
        
        max_attempts = kwargs.get("max_attempts", 5)
        rate_limit_time = kwargs.get("rate_limit_time", 30)
        
        response = None
        attempt_n = 0
        while response is None and attempt_n < max_attempts:
            try:
                response = client.messages.create(**request_load)
            except (BadRequestError, InternalServerError) as e:
                attempt_n += 1
                log.exception(f"Attempt {attempt_n}: Got error - {e}")
                time.sleep(r.uniform(0.5, 2.0))
            except RateLimitError as e:
                attempt_n += 1
                log.exception(f"Attempt {attempt_n}: Got RateLimit error - {e}")
                time.sleep(rate_limit_time)
        
            if isinstance(response, Message):
                content = next(i.input if "tool_use" in i.type else i.text for i in response.content)
                request_out = self._request_out(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    user=user,
                    system=system,
                    content=content,
                    schema=schema
                )
            else:
                log.warning(
                    f"Response was not a Message instance. Got - {response}"
                )
                return response
        
        if attempt_n == max_attempts:
            log.error("Max attempts exceeded.")
            raise
            
        if self.print_response:
            print(f"Request response: {request_out.text}")
            print(f"Request meta: {request_out.meta}")
        
        return request_out
        