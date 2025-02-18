from pydantic import BaseModel
from llms.vertexai_client import VertexAIClient


class GeminiConfigs(BaseModel):
    pass


class Gemini(VertexAIClient):
    pass