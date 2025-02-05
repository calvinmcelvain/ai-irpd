import os
import re
import importlib
import json
import logging
from pathlib import Path
from dotenv import load_dotenv 
from json import JSONDecodeError
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)


def get_env_var(key: str) -> str:
    """
    Retrieve the environment variable associated with `key`.
    """
    value = os.getenv(key)
    path = Path(__file__).resolve().parent.parent / "configs" / "configs.env"
    if value is None:
        load_dotenv(path, override=True)
        value = os.getenv(key)
        if value is None:
            log.warning(
                f"Environment variable '{key}' is missing or set to NA."
            )
    
    return value


def lazy_import(module_name, class_name):
    """
    Function found from:
    https://github.com/TIGER-AI-Lab/MEGA-Bench/blob/main/megabench/utils.py
    """
    def importer():
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    return importer()


def load_json(file_path: str | Path) -> object:
    """
    Returns the JSON object from a JSON file.
    """
    try:
        json_data = json.loads(Path(file_path).read_text())
    except (JSONDecodeError, FileNotFoundError) as e:
        log.error(f"Error loading JSON from {file_path}: {e}")
        return None
    return json_data


def validate_json(json_data: object, schema: BaseModel) -> object:
    """
    Returns the object from json schema validation.
    """
    try:
        schema_obj = schema.model_validate(json_data)
        return schema_obj
    except ValidationError as e:
        log.error(
            f"Validation error for schema '{schema.__name__}': {e}\n"
            f"JSON data: {json.dumps(json_data, indent=2)}"
        )
        return None


def file_to_string(file_path: str | Path) -> str:
    """
    Return file contents as a string.
    """
    return Path(file_path).read_text()


def write_file(file_path: str | Path, file_write: str) -> None:
    """
    Write a string to a file at the given path.
    """
    Path(file_path).write_text(file_write)


def check_directories(paths: list[str]) -> bool:
    """
    Check if all given directories exist.
    """
    return all(Path(path).is_dir() for path in paths)


def get_nested_attr(obj: object, attr_path: str) -> object:
    """
    Gets nested attribute.
    """
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def regex_group(string: str, pattern: str, group: int = 1) -> str:
    """
    Returns group of regex match.
    """
    match = re.search(pattern, string)
    return match.group(group)