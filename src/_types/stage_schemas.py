"""
Schemas module.

Contains the Pydantic output schemas for all stages. Used for JSON structured 
outputs of LLMs.
"""
from pydantic import BaseModel


# Schemas for stage schemas containing list of outputs (such as categories or
# category assignments/classifications)
class Examples(BaseModel):
    window_number: int
    reasoning: str


class Category(BaseModel):
    category_name: str
    definition: str
    examples: list[Examples]


class CategoryAssignment(BaseModel):
    category_name: str


class Ranking(BaseModel):
    category_name: str
    rank: int


# Stage schemas
class Stage0Schema(BaseModel):
    window_number: int
    summary: str


class Stage1Schema(BaseModel):
    categories: list[Category]


class Stage1rSchema(BaseModel):
    refined_categories: list[Category]


class Stage1cSchema(BaseModel):
    refined_categories: list[Category]


class Stage2Schema(BaseModel):
    window_number: int
    assigned_categories: list[CategoryAssignment]
    reasoning: str


class Stage3Schema(BaseModel):
    window_number: int
    category_ranking: list[Ranking]
    reasoning: str