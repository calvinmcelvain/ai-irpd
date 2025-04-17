"""
Functions module.

Contains useful functions specific to IRPD testing.
"""
import logging
from pathlib import Path
from typing import List
from pydantic import BaseModel

from _types.irpd_output import IRPDOutput
from _types.stage_schemas import Category


log = logging.getLogger("app")



def categories_to_txt(categories: List[Category]) -> str:
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
    

def complete_irpdout(
    stage_name: str,subset_path: Path, irpd_output: IRPDOutput
) -> IRPDOutput:
    """
    Fills the paths in IRPDOutput object.
    """
    prefix = f"{subset_path.name}_"
    system_path = f"{prefix}stg_{stage_name}_system_prompt.txt"
    if stage_name in {"2", "3"}:
        user_path = f"{prefix}{irpd_output.parsed.window_number}_user_prompt.txt"
        response_path = f"{prefix}{irpd_output.parsed.window_number}_response.txt"
    else:
        user_path = f"{prefix}stg_{stage_name}_user_prompt.txt"
        response_path = f"{prefix}stg_{stage_name}_response.txt"
    irpd_output.system_path = subset_path / system_path
    irpd_output.user_path = subset_path / user_path
    irpd_output.response_path = subset_path / response_path
    return irpd_output