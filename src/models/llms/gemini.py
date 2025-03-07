import logging
import time
import random as r
from pydantic import BaseModel, Field
from google import genai
from google.genai.types import GenerateContentConfig, GenerateContentResponse
from google.api_core.exceptions import (
    ResourceExhausted,
    InternalServerError
)
from llms.base_model import Base

log = logging.getLogger(__name__)


class GeminiConfigs(BaseModel):
    max_output_tokens: int = Field(None, ge=1, le=4096)
    temperature: float = Field(None, ge=0, le=1)
    top_p: float = Field(None, ge=0, le=1)
    top_k: float = Field(None, ge=0, le=1)
    seed: int = Field(None, ge=1)
    frequency_penalty: float = Field(None, ge=0, le=1)
    presence_penalty: float = Field(None, ge=0, le=1)


class Gemini(Base):
    
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
    
    def request(self, user, system, schema: BaseModel = None, **kwargs):
        client = self.create_client()
        
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
                request_out = response
                output_tokens = response.usage_metadata.candidates_token_count
                input_tokens = response.usage_metadata.prompt_token_count
                tokens = {"output_tokens": output_tokens, "input_tokens": input_tokens}
                content = response.parsed.model_dump_json()
                request_out = self._process_output(
                    id="none",
                    tokens=tokens, 
                    content=content,
                    system=system,
                    user=user
                )
            else:
                log.error(f"Response was not a Message instance. Got - {response}")
                return response

        if self.print_response:
            print(f"Request response: {request_out.response}")
            print(f"Request meta: {request_out.meta}")
        
        return request_out