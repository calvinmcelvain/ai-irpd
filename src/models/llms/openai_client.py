import time
import logging
import random as r
from abc import abstractmethod
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional, Union
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError
from openai.types.chat import ChatCompletion

from models.batch_out import BatchOut, BatchResponse
from models.prompts import Prompts
from models.llms.base_llm import BaseLLM


log = logging.getLogger(__name__)



class OpenAIToolCall(BaseModel):
    name: str = "json_response"
    parameters: object | None


class OpenAIClient(BaseLLM):
    def __init__(
        self,
        api_key = None,
        model = None,
        configs = None,
        print_response = False,
        json_tool = False,
        base_url: str = None,
        **kwargs
    ):
        super().__init__(
            api_key,
            model,
            configs,
            print_response,
            json_tool,
            **kwargs
        )
        self.base_url = base_url
    
    def create_client(self):
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    @abstractmethod
    def default_configs(self):
        pass
    
    def _prep_messages(self, user: str, system: str):
        return {"messages": [self._prep_system_message(system), self._prep_user_message(user)]}
    
    def _json_tool_call(self, schema: BaseModel):
        tool_load = OpenAIToolCall(parameters=schema.model_json_schema())
        tool_choice = {"name": "json_output", "type": "tool"}
        return {"tools": [tool_load.model_dump()], "tool_choice": tool_choice}
    
    def _request_load(
        self,
        user: str,
        system: str,
        schema: Optional[BaseModel],
        schema_dumps: bool = False
    ):
        request_load = {"model": self.model}
        request_load.update(self.configs.model_dump(exclude_none=True))
        request_load.update(self._prep_messages(user, system))
        request_load.update(self._json_tool_call(schema)) if self.json_tool else {}
        
        schema = schema.model_json_schema() if schema_dumps else schema
        request_load.update({"response_format": schema}) if schema else {}
        return request_load
        
    def format_batch(self, messages: List[Prompts], message_ids: List[str], schema: BaseModel = None):
        batch = []
        for idx, message in enumerate(messages):
            user = message.user
            system = message.system
            batch_input = {"custom_id": message_ids[idx], "method": "POST", "url": "/v1/chat/completions"}
            request_load = self._request_load(
                user=user, system=system, schema=schema, schema_dumps=True
            )
            batch_input.update({"body": request_load})
            batch.append(batch_input)
        return batch
    
    def batch_request(self, batch_file: Union[str, Path]):
        client = self.create_client()
        
        file = client.files.create(file=open(batch_file, "rb"), purpose="batch")
        
        batch = client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        if batch.errors:
            log.error(f"Error in batch request: {batch.errors.model_dump_json()}")
        return batch.id
    
    def _get_batch_id(self, batch_file: Union[str, Path]):
        client = self.create_client()
        files = client.files.list()
        batches = client.batches.list()
        file_id = next((f.id for f in files.data if f.filename == batch_file))
        batch_id = next((b.id for b in batches.data if b.input_file_id == file_id))
        return batch_id
    
    def retreive_batch(self, batch_id: str, schema: BaseModel):
        client = self.create_client()
        
        batch = client.batches.retrieve(batch_id=batch_id)
        if batch.status != "completed":
            return batch.status
        
        batch_output = client.files.content(file_id=batch.output_file_id)
        batch_input = client.files.content(file_id=batch.input_file_id)
        
        output = BatchOut(
            batch_id=batch_id,
            responses=[]
        )
        
        for response in batch_output:
            response_id = response["custom_id"]
            
            prompts = next((p["body"]["messages"] for p in batch_input if p["custom_id"] == response_id))
            system = next((p["content"] for p in prompts if p["role"] != "user"))
            user = next((p["content"] for p in prompts if p["role"] == "user"))
            
            response_data = response["response"]["body"]
            output.responses.append(self._request_out(
                input_tokens=response_data["usage"]["input_tokens"],
                output_tokens=response_data["usage"]["completion_tokens"],
                system=system,
                user=user,
                content=response_data["message"]["content"],
                schema=schema
            ))
        return output
    
    def request(self, user: str, system: str, schema: BaseModel = None, **kwargs):
        client = self.create_client()
        
        request_load = self._request_load(
            user=user,
            system=system,
            schema=schema
        )
        
        max_attempts = kwargs.get("max_attempts", 5)
        rate_limit_time = kwargs.get("rate_limit_time", 30)
        
        response = None
        attempt_n = 0
        while response is None and attempt_n < max_attempts:
            try:
                if schema and not self.json_tool:
                    response = client.beta.chat.completions.parse(**request_load)
                else:
                    response = client.chat.completions.create(**request_load)
            except (APIConnectionError, APITimeoutError) as e:
                attempt_n += 1
                log.exception(f"Attempt {attempt_n}: Got error - {e}")
                time.sleep(r.uniform(0.5, 2.0))
            except RateLimitError as e:
                attempt_n += 1
                log.exception(f"Attempt {attempt_n}: Got RateLimit error - {e}")
                time.sleep(rate_limit_time)
        
            if isinstance(response, ChatCompletion):
                request_out = self._request_out(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    system=system,
                    user=user,
                    content=response.choices[0].message.content,
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