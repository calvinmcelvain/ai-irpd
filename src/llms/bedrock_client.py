import logging
import time
import random as r
import json
import boto3
from pydantic import BaseModel
from llms.base_model import Base
from abc import abstractmethod

log = logging.getLogger(__name__)


class BedrockToolCall(BaseModel):
    name: str = "json_response"
    inputSchema: object | None


class BedrockClient(Base):
    def __init__(
        self,
        api_key = None,
        model = None,
        configs = None,
        print_response = False,
        json_tool = False,
        region: str = "us-east-1",
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
        self.region = region
    
    @abstractmethod
    def default_configs(self):
        pass
    
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
        user_m = user + "/n/n" + "Use the json_response tool."
        return user_m
    
    def _prep_messages(self, user: str, system: str):
        messages = {"system": [{"text": system}]}
        messages.update({"messages": [self._prep_user_message(user)]})
        return messages
    
    def _dump_response(self, response: dict):
        content_json = json.loads(response.get('body').read())
        content_out = content_json["output"]["message"]["content"]
        out = next(i["toolUse"]["input"] for i in content_out if "toolUse" in i)
        return out
    
    def request(self, user: str, system: str, schema: BaseModel = None, **kwargs):
        client = self.create_client()
        
        user_m = self._add_json_requirement(user) if schema else user
        body_load = self._prep_messages(user_m, system)
        body_load.update({"inferenceConfig": self.configs.model_dump(exclude_none=True)})
        body_load.update(self._json_tool_call(schema)) if schema else {}
        
        request_load = {"modelId": self.model}
        request_load.update({"contentType": "application/json"})
        request_load.update({"body": json.dumps(body_load)})
        
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
            id = meta_data['RequestId']
            output_tokens = int(meta_data['HTTPHeaders']['x-amzn-bedrock-output-token-count'])
            input_tokens = int(meta_data['HTTPHeaders']['x-amzn-bedrock-input-token-count'])
            tokens = {"output_tokens": output_tokens, "input_tokens": input_tokens}
            content = self._dump_response(response)
            request_out = self._process_output(
                id=id,
                tokens=tokens,
                content=content,
                system=system,
                user=user
            )

        if self.print_response:
            print(f"Request response: {request_out.response}")
            print(f"Request meta: {request_out.meta}")
        
        return request_out
        
        