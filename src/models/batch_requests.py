import logging
from typing import List

from models.llms.base_llm import BaseLLM


log = logging.getLogger(__name__)



class BatchRequest:
    def __init__(self, llm: BaseLLM, messages: List[str]):
        self.llm = llm
        self.messages = messages

    def prepare_batch(self):
        return self.llm.format_batch(self.messages)

    async def send_batch(self):
        formatted_batch = self.prepare_batch()
        return await self.llm.batch_request(formatted_batch)