from pydantic import BaseModel


class Stage0Schema(BaseModel):
    window_number: int
    summary: str


class Examples(BaseModel):
    window_number: int
    reasoning: str


class Category(BaseModel):
    category_name: str
    definition: str
    examples: list[Examples]


class Stage1Schema(BaseModel):
    categories: list[Category]


class Stage1rSchema(BaseModel):
    refined_categories: list[Category]


class CategoryAssignment(BaseModel):
    category_name: str


class Stage2Schema(BaseModel):
    window_number: int
    assigned_categories: list[CategoryAssignment]
    reasoning: str


class Ranking(BaseModel):
    category_name: str
    rank: int


class Stage3Schema(BaseModel):
    window_number: int
    category_ranking: list[Ranking]
    reasoning: str