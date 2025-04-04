import logging
from typing import List
from pydantic import BaseModel


log = logging.getLogger(__name__)



def categories_to_txt(categories: BaseModel) -> str:
    """
    Function that takes categories, and outputs format for pdf/prompt.
    """
    category_texts = []
    for category in categories:
        example_texts = []
        for idx, example in enumerate(category.examples, start=1):
            example_texts.append(
                f"  {idx}. Window number: {example.window_number},"
                f" Reasoning: {example.reasoning}"
            )
        category_text = (
            f"### {category.category_name}\n\n"
            f"**Definition**: {category.definition}\n\n"
            f"**Examples**:\n\n{"\n".join(example_texts)}\n\n"
        )
        category_texts.append(category_text)
    return "".join(category_texts)


def instance_types(case: str) -> List[str]:
    """
    Function that returns list of instance types for a given case.
    """
    if case in {"uni", "uniresp"}:
        return ["ucoop", "udef"]
    return ["coop", "def"]


def output_attrb(output: BaseModel) -> BaseModel:
    """
    Returns the output for a given output
    """
    # Stage 1
    if hasattr(output, "categories"):
        return output.categories
    # Stage 1r & 1c
    if hasattr(output, "refined_categories"):
        return output.refined_categories
    # Stage 2
    if hasattr(output, "assigned_categories"):
        return output.assigned_categories
    # Stage 3
    if hasattr(output, "category_ranking"):
        return output.category_ranking