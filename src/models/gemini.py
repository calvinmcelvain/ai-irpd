from pydantic import BaseModel
from models.vertexai_client import VertexAIClient


class GeminiConfigs(BaseModel):
    pass


class Gemini(VertexAIClient):
    pass