import time
import logging
import random as r
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from models.base_model import Base, RequestOut
from abc import abstractmethod

log = logging.getLogger(__name__)


class OpenAIToolCall(BaseModel):
    name: str = "json_response"
    parameters: object | None


class OpenAIClient(Base):
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
    
    def request(self, user: str, system: str, schema: BaseModel = None, **kwargs):
        client = self.create_client()
        
        request_load = {"model": self.model}
        request_load.update(self.configs.model_dump(exclude_none=True))
        request_load.update(self._prep_messages(user, system))
        request_load.update(self._json_tool_call(schema)) if self.json_tool else {}
        request_load.update({"response_format": schema}) if schema else {}
        
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
                log.info(
                    f"Attempt {attempt_n}: Got error - {e}"
                )
                time.sleep(r.uniform(0.5, 2.0))
            except RateLimitError as e:
                attempt_n += 1
                log.info(
                    f"Attempt {attempt_n}: Got RateLimit error - {e}"
                )
                time.sleep(rate_limit_time)
        
            if isinstance(response, ChatCompletion):
                request_out = RequestOut(
                    response=response.choices[0].message.content,
                    meta=response
                )
            else:
                log.info(
                    f"Response was not a Message instance. Got - {response}"
                )

        if self.print_response:
            log.info(
                f"Request response: {request_out.response}"
            )
            log.info(
                f"Request meta: {request_out.meta}"
            )
        
        return request_out