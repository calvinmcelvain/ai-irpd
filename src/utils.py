"""
Utilities module.

Contains general functions.
"""
import os
import subprocess
import sys
import re
import importlib
import json
import yaml
import logging
import configs
import importlib.resources as pkg_resources
from typing import List, Union
from pathlib import Path
from dotenv import load_dotenv 
from yaml import YAMLError
from json import JSONDecodeError
from markdown_pdf import MarkdownPdf, Section
from pydantic import BaseModel, ValidationError


log = logging.getLogger(__name__)



def to_list(arg: Union[object, List[object]]):
    """
    Returns listed arg.
    """
    if isinstance(arg, list):
        return arg
    return [arg]


def get_env_var(key: str) -> str:
    """
    Retrieve the environment variable associated with `key`.
    """
    value = os.getenv(key)
    path = Path().resolve().home() / "dotfiles" / "irpd_configs.env"
    if value is None:
        load_dotenv(path, override=True)
        value = os.getenv(key)
        if value is None:
            log.warning(
                f"Environment variable '{key}' is missing or set to NA."
            )
    return value


def create_directory(paths: List[Union[str, Path]]) -> None:
    """
    Creates directory for paths.
    """
    paths = to_list(paths)
    for path in paths:
        path_c = Path(path)
        try:
            if not path_c.exists():
                path_c.mkdir(exist_ok=True, parents=True)
                log.info(f"PATH: Created directory: {path_c.as_posix()}")
            else:
                log.info(f"PATH: Path exists: {path_c.as_posix()}")
        except Exception as e:
            log.error(f"Error creating directory '{path_c}': {e}")
            raise
    return None


def lazy_import(module_name: str, class_name: str) -> object:
    """
    Function found from:
    https://github.com/TIGER-AI-Lab/MEGA-Bench/blob/main/megabench/utils.py
    """
    def importer():
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    return importer()


def load_config(config: str) -> dict:
    """
    Load a YAML or JSON config file.
    """
    if not config.strip():
        raise ValueError("Config filename cannot be empty.")

    if not config.endswith((".yml", ".yaml", ".json")):
        log.warning(
            f"Config file '{config}' missing .yml, .yaml, or .json extension."
            " Appending .yml extension by default."
        )
        config += ".yml"

    try:
        with pkg_resources.open_text(configs, config) as f:
            if config.endswith((".yml", ".yaml")):
                return yaml.safe_load(f)
            elif config.endswith(".json"):
                return json.load(f)
    except FileNotFoundError:
        log.error(f"Configuration file not found: {config}")
        raise
    except YAMLError as e:
        log.error(f"Error parsing YAML file '{config}': {e}")
        raise
    except JSONDecodeError as e:
        log.error(f"Error parsing JSON file '{config}': {e}")
        raise


def load_json(file_path: Union[str, Path], dumps: bool = False) -> dict | str:
    """
    Returns the JSON object (or string) from a JSON file.
    """
    try:
        json_data = json.loads(Path(file_path).read_text())
    except (JSONDecodeError, FileNotFoundError) as e:
        log.error(f"Error loading JSON from {file_path}: {e}")
        raise
    return json.dumps(json_data) if dumps else json_data


def load_jsonl(file_path: Union[str, Path], dumps: bool = False) -> List[dict] | str:
    """
    Returns a list of JSON objects (or a JSON string) from a JSONL file.
    """
    try:
        with open(Path(file_path), "r") as f:
            json_data = [json.loads(line) for line in f if line.strip()]
    except (JSONDecodeError, FileNotFoundError) as e:
        log.error(f"Error loading JSONL from {file_path}: {e}")
        raise
    return json.dumps(json_data) if dumps else json_data


def validate_json(json_data: dict, schema: BaseModel) -> BaseModel | None:
    """
    Returns the object from json schema validation.
    """
    try:
        schema_obj = schema.model_validate(json_data)
        return schema_obj
    except ValidationError as e:
        log.exception(
            f"Validation error for schema '{schema.__name__}': {e}\n"
            f"JSON data: {json.dumps(json_data, indent=2)}"
        )
        return None


def load_json_n_validate(
    file_path: Union[str, Path],
    schema: BaseModel
) -> BaseModel | None:
    """
    Loads json file and validates for schema.
    """
    json_data = load_json(file_path)
    return validate_json(json_data, schema)


def validate_json_string(json_str: str, schema: BaseModel) -> BaseModel | None:
    """
    Returns the object from json schema validation.
    """
    try:
        schema_obj = schema.model_validate_json(json_str)
        return schema_obj
    except Exception as e:
        log.exception(f"Error in model validation': {e}\n")
        return None


def file_to_string(file_path: Union[str, Path]) -> str:
    """
    Return file contents as a string.
    """
    try:
        return Path(file_path).read_text()
    except FileNotFoundError:
        log.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        log.error(f"Error reading file '{file_path}': {e}")
        raise


def write_file(
    file_paths: Union[Union[str, Path], List[Union[str, Path]]],
    file_writes: Union[str, List[str]]
) -> None:
    """
    Write a list of strings to specified file paths.
    """
    file_paths = to_list(file_paths)
    file_writes = to_list(file_writes)
    assert len(file_paths) == len(file_writes), "`file_paths` and `file_writes` must be same length."
    for idx, path in enumerate(file_paths):
        path = Path(path)
        try:
            path.write_text(file_writes[idx])
        except Exception as e:
            log.error(f"Error writing to file '{path.as_posix()}': {e}")
            raise
        

def write_jsonl(
    file_path: Union[str, Path],
    json_obj: Union[dict, List[dict]]
):
    """
    Writes jsonl file from json/dict object.
    """
    json_obj = to_list(json_obj)
    write_file(file_paths=file_path, file_writes="")
    with open(Path(file_path), "w") as f:
        for line in json_obj:
            json.dump(line, f)
            f.write("\n")


def write_json(file_path: Union[str, Path], data: dict, indent: int = 4) -> None:
    """
    Write JSON data to a file at the given path.
    """
    try:
        Path(file_path).write_text(json.dumps(data, indent=indent))
    except TypeError as e:
        log.error(f"Error serializing JSON data: {e}")
        raise
    except Exception as e:
        log.error(f"Error writing JSON to file '{file_path}': {e}")
        raise


def check_directories(paths: Union[Union[str, Path], List[Union[str, Path]]]) -> bool:
    """
    Check if all given directories exist.
    """
    try:
        paths = to_list(paths)
        return all(Path(path).is_dir() for path in paths)
    except Exception as e:
        log.exception(f"Error checking directories: {e}")
        return False


def find_named_parent(path: Path, target: str) -> Path | None:
    """
    Finds the nearest parent directory with the given name.
    """
    try:
        for parent in path.parents:
            if parent.name == target:
                return parent
    except Exception as e:
        log.exception(
            f"Error finding named parent '{target}' in path '{path}': {e}"
        )
        return None


def get_nested_attr(obj: object, attr_path: str) -> object:
    """
    Gets nested attribute.
    """
    try:
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj
    except AttributeError as e:
        log.error(
            f"Error accessing attribute '{attr}' in path '{attr_path}': {e}"
        )
        raise


def regex_group(string: str, pattern: str, group: int = 1) -> str:
    """
    Returns group of regex match.
    """
    try:
        match = re.search(pattern, string)
        if match:
            return match.group(group)
        else:
            log.warning(
                f"No match found for pattern '{pattern}' in string '{string}'"
            )
            return ""
    except IndexError:
        log.exception(
            f"Group {group} not found in the match for pattern '{pattern}'"
        )
        return ""
    except re.error as e:
        log.exception(
            f"Regex error for pattern '{pattern}': {e}"
        )
        return ""


def txt_to_pdf(text: str, file_path: Path) -> None:
    """
    Saves text as pdf to file path.
    """
    pdf = MarkdownPdf()
    pdf.add_section(Section(text))
    pdf.save(file_path)
    

def is_tail_running() -> bool:
    """
    Checks to see if terminal is running a tail log.
    """
    if sys.platform == "win32":
        result = subprocess.run([
            "tasklist"
        ], capture_output=True, text=True)
        return "tail" in result.stdout
    else:
        result = subprocess.run([
            "pgrep",
            "-f",
            "tail -f logs/app.log"
        ], capture_output=True, text=True)
        return result.returncode == 0