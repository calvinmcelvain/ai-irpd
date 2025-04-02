import time
import logging
import json
import random as r
from abc import abstractmethod
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional, Tuple
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError
from openai.types.chat import ChatCompletion
from openai.lib._parsing._completions import type_to_response_format_param

from utils import write_jsonl, load_jsonl
from models.batch_output import BatchOut, BatchResponse
from models.prompts import Prompts
from models.llms.base_llm import BaseLLM


log = logging.getLogger(__name__)



class OpenAIToolCall(BaseModel):
    name: str = "json_response"
    parameters: object | None


class OpenAIClient(BaseLLM):
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
        schema: Optional[BaseModel]
    ):
        request_load = {"model": self.model}
        request_load.update(self.configs.model_dump(exclude_none=True))
        request_load.update(self._prep_messages(user, system))
        request_load.update(self._json_tool_call(schema)) if self.json_tool else {}
        request_load.update({"response_format": schema}) if schema else {}
        return request_load
        
    def _format_batch(
        self,
        messages: List[Tuple[str, Prompts]],
        schema: BaseModel = None
    ):
        batch = []
        for message_id, message in messages:
            user = message.user
            system = message.system
            batch_input = {"custom_id": message_id, "method": "POST", "url": "/v1/chat/completions"}
            schema = type_to_response_format_param(schema)
            request_load = self._request_load(user, system, schema)
            batch_input.update({"body": request_load})
            batch.append(batch_input)
        return batch
    
    def retreive_batch(
        self,
        batch_id: str,
        schema: Optional[BaseModel] = None,
        batch_file_path: Optional[Path] = None
    ):
        client = self.create_client()
        
        batch = client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            log.info(f"Batch {batch_id} is {batch.status}.")
            return None
        
        batch_output_file = client.files.content(file_id=batch.output_file_id).iter_lines()
        batch_input_file = load_jsonl(batch_file_path) if batch_file_path else None
        
        batch_output = BatchOut(
            batch_id=batch_id,
            responses=[]
        )
        for response in batch_output_file:
            response_json = json.loads(response)
            response_id = response_json["custom_id"]
            
            if batch_input_file:
                prompts = next((
                    p["body"]["messages"] 
                    for p in batch_input_file if p["custom_id"] == response_id
                ))
                system = next((p["content"] for p in prompts if p["role"] != "user"))
                user = next((p["content"] for p in prompts if p["role"] == "user"))
            else:
                system, user = "None"
            
            response_data = response_json["response"]["body"]
            request_out = self._request_out(
                input_tokens=response_data["usage"]["prompt_tokens"],
                output_tokens=response_data["usage"]["completion_tokens"],
                system=system,
                user=user,
                content=response_data["choices"][0]["message"]["content"],
                schema=schema
            )
            batch_output.responses.append(BatchResponse(
                response_id=response_id,
                response=request_out
            ))
        return batch_output
    
    def request_batch(
        self,
        messages: List[Tuple[str, Prompts]],
        schema: Optional[BaseModel] = None,
        batch_file_path: Optional[Path] = None
    ):
        client = self.create_client()
        
        formatted_batch = self._format_batch(messages, schema)
        
        save_batch = True if batch_file_path else False
        
        batch_file_path = Path(batch_file_path or "/tmp/formatted_batch.jsonl")
        write_jsonl(batch_file_path, formatted_batch)
        file = client.files.create(file=batch_file_path.open("rb"), purpose="batch")
        
        if not save_batch: batch_file_path.unlink()
        
        batch = client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        if batch.errors:
            log.error(f"Error in batch request: {batch.errors.model_dump_json()}")
        
        return batch.id
    
    def request(self, user: str, system: str, schema: BaseModel = None, **kwargs):
        client = self.create_client()
        
        request_load = self._request_load(user, system, schema)
        
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