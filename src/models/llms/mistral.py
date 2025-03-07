import logging
import time
import random as r
import mistralai
from mistralai import ChatCompletionResponse
from pydantic import BaseModel, Field
from pydantic import BaseModel
from llms.base_model import Base

log = logging.getLogger(__name__)

class MistralConfigs(BaseModel):
    max_completion_tokens: int = Field(None, ge=1, le=4096)
    temperature: float = Field(None, ge=0, le=1)
    top_p: float = Field(None, ge=0, le=1)
    random_seed: int = Field(None, ge=1)
    frequency_penalty: float = Field(None, ge=0, le=1)
    presence_penalty: float = Field(None, ge=0, le=1)


class MistralToolCall(BaseModel):
    name: str = "json_response"
    parameters: object | None


class Mistral(Base):
    
    def create_client(self):
        return mistralai.Mistral(api_key=self.api_key)
    
    def default_configs(self):
        return MistralConfigs()
    
    def _prep_messages(self, user: str, system: str):
        return {"messages": [self._prep_system_message(system), self._prep_user_message(user)]}
    
    def _json_tool_call(self, schema: BaseModel):
        tool_load = MistralToolCall(parameters=schema.model_json_schema())
        tool_choice = {"name": "json_output", "type": "tool"}
        return {"tools": [tool_load.model_dump()], "tool_choice": tool_choice}
    
    def request(self, user: str, system: str, schema: BaseModel = None, **kwargs):
        client = self.create_client()
        
        request_load = {"model": self.model}
        request_load.update(self.configs.model_dump(exclude_none=True))
        request_load.update(self._prep_messages(user, system))
        request_load.update(self._json_tool_call(schema)) if self.json_tool else {}
        request_load.update({"response_format": schema}) if schema and not self.json_tool else ()
        
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
                id = response.id
                output_tokens = response.usage.completion_tokens
                input_tokens = response.usage.prompt_tokens
                tokens = {"output_tokens": output_tokens, "input_tokens": input_tokens}
                content = response.choices[0].message.content
                request_out = self._process_output(
                    id=id,
                    tokens=tokens, 
                    content=content,
                    system=system,
                    user=user
                )
            else:
                log.warning(
                    f"Response was not a Message instance. Got - {response}"
                )
                return response

        if self.print_response:
            print(f"Request response: {request_out.response}")
            print(f"Request meta: {request_out.meta}")
        
        return request_out