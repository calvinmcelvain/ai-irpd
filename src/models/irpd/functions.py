import logging
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