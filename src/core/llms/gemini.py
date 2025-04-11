"""
Google's Gemini module.

Contains Gemini model and configs.
"""

import logging
import time
import random as r
from typing import Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai.types import GenerateContentConfig, GenerateContentResponse
from google.api_core.exceptions import ResourceExhausted, InternalServerError

from types.prompts import Prompts
from models.llms.base_llm import BaseLLM


log = logging.getLogger(__name__)



class GeminiConfigs(BaseModel):
    max_output_tokens: int = Field(None, ge=1, le=4096)
    temperature: float = Field(None, ge=0, le=1)
    top_p: float = Field(None, ge=0, le=1)
    top_k: float = Field(None, ge=0, le=1)
    seed: int = Field(None, ge=1)
    frequency_penalty: float = Field(None, ge=0, le=1)
    presence_penalty: float = Field(None, ge=0, le=1)


class Gemini(BaseLLM):
    """
    Google's Gemini class.
    
    Defines request methods for genai SDK.
    """
    def create_client(self):
        return genai.Client(api_key=self.api_key)
    
    def default_configs(self):
        return GeminiConfigs()
    
    @staticmethod
    def _prep_system_message(system: str):
        return {"system_instruction": system}
    
    @staticmethod
    def _prep_user_message(user: str):
        return {"contents": user}
    
    def _request_load(
        self,
        user: str,
        system: str,
        schema: Optional[BaseModel]
    ):
        configs = self.configs.model_dump(exclude_none=True)
        configs.update(self._prep_system_message(system))
        if schema:
            configs.update({
                "response_mime_type": "application/json",
                "response_schema": schema
            })
        request_load = {"model": self.model}
        request_load.update(self._prep_user_message(user))
        request_load.update({"config": GenerateContentConfig(**configs)})
        return request_load
    
    def _format_batch(self, messages, schema = None):
        pass
    
    def retreive_batch(self, batch_id, schema = None, batch_file_path = None):
        pass
    
    def request_batch(self, messages, schema = None, batch_file_path = None):
        pass
    
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
        while response is None and attempt_n < max_attempts:
            try:
                response = client.models.generate_content(**request_load)
            except ResourceExhausted:
                attempt_n += 1
                log.exception(
                    f"Quota Exhausted. Waiting {rate_limit_time} seconds before retry..."
                )
                time.sleep(rate_limit_time)
            except InternalServerError:
                attempt_n += 1
                log.exception("API error. Retrying in a few seconds...")
                time.sleep(r.uniform(0.5, 2))
            except ValueError as e:
                attempt_n += 1
                log.exception(f"Attempt {attempt_n}: Got error - {e}")
                time.sleep(r.uniform(0.5, 2.0))
        
            if isinstance(response, GenerateContentResponse):
                content = response.parsed.model_dump_json()
                request_out = self._request_out(
                    input_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count,
                    system=system,
                    user=user,
                    content=content,
                    schema=schema
                )
            else:
                log.error(f"Response was not a Message instance. Got - {response}")
                return response

        if attempt_n == max_attempts:
            log.error("Max attempts exceeded.")
            raise
        
        if self.print_response:
            print(f"Request response: {request_out.response}")
            print(f"Request meta: {request_out.meta}")
        
        return request_out