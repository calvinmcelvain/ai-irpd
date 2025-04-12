"""
Mistral's API client module.

Defines the MistralClient model.
"""
import logging
import json
import time
import random as r
import mistralai
from pathlib import Path
from typing import Optional, List, Tuple
from mistralai import ChatCompletionResponse
from pydantic import BaseModel
from openai.lib._parsing._completions import type_to_response_format_param

from helpers.utils import write_jsonl, load_jsonl
from types.batch_output import BatchOut, BatchResponse
from types.prompts import Prompts
from core.llms.clients.base import BaseLLM


log = logging.getLogger(__name__)



class MistralToolCall(BaseModel):
    name: str = "json_response"
    parameters: object | None


class MistralClient(BaseLLM):
    """
    Mistral class.
    
    Defines request methods using the Mistral API.
    """
    def create_client(self):
        return mistralai.Mistral(api_key=self.api_key)
    
    def _translate_config(self, config):
        # Defined at Model-level
        return super()._translate_config(config)
    
    def _prep_messages(self, user: str, system: str):
        return {"messages": [self._prep_system_message(system), self._prep_user_message(user)]}
    
    def _json_tool_call(self, schema: BaseModel):
        tool_load = MistralToolCall(parameters=schema.model_json_schema())
        tool_choice = {"name": "json_output", "type": "tool"}
        return {"tools": [tool_load.model_dump()], "tool_choice": tool_choice}
    
    def _request_load(
        self,
        user: str,
        system: str,
        schema: Optional[BaseModel]
    ):
        request_load = {"model": self.model}
        request_load.update(self.configs)
        request_load.update(self._prep_messages(user, system))
        request_load.update(self._json_tool_call(schema)) if self.json_tool else {}
        request_load.update({"response_format": schema}) if schema and not self.json_tool else {}
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
            batch_input = {"custom_id": message_id}
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
        
        batch = client.batch.jobs.get(batch_id)
        
        if batch.status != "SUCCESS":
            log.info(f"Batch {batch_id} is {batch.status}.")
            return None
        
        batch_output_file = client.files.download(file_id=batch.output_file)
        batch_input_file = load_jsonl(batch_file_path) if batch_file_path else None
        
        batch_output = BatchOut(
            batch_id=batch_id,
            responses=[]
        )
        for response in batch_output_file:
            response_json = json.loads(response)
            response_id = response_json["custom_id"]
            
            # Matching prompts from batch file to resonse by request IDs.
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
        file = client.files.upload(
            file={
                "file_name": batch_file_path.name,
                "content": batch_file_path.open("rb")
            },
            purpose="batch"
        )
        
        if not save_batch: batch_file_path.unlink()
        
        batch = client.batch.jobs.create(
            input_files=[file.id],
            model=self.model,
            endpoint="/v1/chat/completions"
        )
        
        if batch.errors:
            log.error(f"Error in batch request: {batch.errors.model_dump_json()}")
        
        return batch.id
    
    def request(self, prompts: Prompts, schema: BaseModel = None, **kwargs):
        client = self.create_client()
        
        user = prompts.user
        system = prompts.system
        request_load = self._request_load(user, system, schema)
        
        max_attempts = kwargs.get("max_attempts", 5)
        
        response = None
        attempt_n = 0
        while response is None and attempt_n < max_attempts:
            try:
                if schema and not self.json_tool:
                    response = client.chat.parse(**request_load)
                else:
                    response = client.chat.complete(**request_load)
            except Exception as e:
                attempt_n += 1
                log.exception(f"Attempt {attempt_n}: Got error - {e}")
                time.sleep(r.uniform(0.5, 2.0))
        
            if isinstance(response, ChatCompletionResponse):
                content = response.choices[0].message.content
                request_out = self._request_out(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
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