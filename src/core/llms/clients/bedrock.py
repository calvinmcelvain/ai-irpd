"""
Amazon Bedrock client module.

Contains the BedrockClient model.
"""

import logging
import time
import random as r
import json
import boto3
from typing import Optional
from pydantic import BaseModel

from types.prompts import Prompts
from core.llms.base import BaseLLM


log = logging.getLogger(__name__)



class BedrockToolCall(BaseModel):
    name: str = "json_response"
    inputSchema: object | None


class BedrockClient(BaseLLM):    
    """
    Bedrock client class.

    Defines request methods using the Bedrock client.
    """
    def _translate_config(self, config):
        # Deinfed at the Model-level
        return super()._translate_config(config)
    
    def create_client(self):
        return boto3.client("bedrock-runtime", region_name=self.region)
    
    @staticmethod
    def _prep_user_message(user: str):
        return {"role": "user", "content": [{"text": user}]}
    
    def _json_tool_call(self, schema: BaseModel):
        tool_load = BedrockToolCall(inputSchema={"json": schema.model_json_schema()})
        return {"toolConfig": {"tools": [{"toolSpec": tool_load.model_dump()}]}}
    
    @staticmethod
    def _add_json_requirement(user: str):
        """
        Adds json tool requirement to prompt. Needed for structured outputs for
        models that use tool_call instead.
        """
        user_m = user + "/n/n" + "Use the json_response tool."
        return user_m
    
    def _prep_messages(self, user: str, system: str):
        messages = {"system": [{"text": system}]}
        messages.update({"messages": [self._prep_user_message(user)]})
        return messages
    
    def _dump_response(self, response: dict):
        """
        Returns the insanely dificult response from Bedrock model outputs.
        """
        content_json = json.loads(response.get('body').read())
        content_out = content_json["output"]["message"]["content"]
        out = next(i["toolUse"]["input"] for i in content_out if "toolUse" in i)
        return out
    
    def _format_batch(self, messages, schema = None):
        # No support for batches
        pass
    
    def request_batch(self, messages, schema = None, batch_file_path = None):
        # No support for batches
        pass
    
    def retreive_batch(self, batch_id, schema = None, batch_file_path = None):
        # No support for batches
        pass
    
    def _request_load(
        self,
        user: str,
        system: str,
        schema: Optional[BaseModel]
    ):
        user_m = self._add_json_requirement(user) if schema else user
        body_load = self._prep_messages(user_m, system)
        body_load.update({"inferenceConfig": self.configs})
        body_load.update(self._json_tool_call(schema)) if schema else {}
        request_load = {"modelId": self.model}
        request_load.update({"contentType": "application/json"})
        request_load.update({"body": json.dumps(body_load)})
        return request_load
    
    def request(
        self,
        prompts: Prompts,
        schema: BaseModel = None,
        **kwargs
    ):
        client = self.create_client()
        
        user = prompts.user
        system = prompts.system
        
        request_load = self._request_load(user, system, schema)
        
        max_attempts = kwargs.get("max_attempts", 5)
        rate_limit_time = kwargs.get("rate_limit_time", 30)
        
        response = None
        attempt_n = 0
        ex = client.exceptions
        while response is None and attempt_n < max_attempts:
            try:
                response = client.invoke_model(**request_load)
            except (ex.ModelTimeoutException, ex.ModelErrorException) as e:
                attempt_n += 1
                log.exception(f"Attempt {attempt_n}: Got error - {e}")
                time.sleep(r.uniform(0.5, 2.0))
            except (ex.ThrottlingException, ex.ServiceQuotaExceededException) as e:
                attempt_n += 1
                log.exception(f"Attempt {attempt_n}: Got RateLimit error - {e}")
                time.sleep(rate_limit_time)
            meta_data = response['ResponseMetadata']
            content = self._dump_response(response)
            request_out = self._request_out(
                input_tokens=int(meta_data['HTTPHeaders']['x-amzn-bedrock-input-token-count']),
                output_tokens=int(meta_data['HTTPHeaders']['x-amzn-bedrock-output-token-count']),
                user=user,
                system=system,
                content=content,
                schema=schema
            )
        
        if attempt_n == max_attempts:
            log.error("Max attempts exceeded.")
            raise

        if self.print_response:
            print(f"Request response: {request_out.text}")
            print(f"Request meta: {request_out.meta}")
        
        return request_out
        
        