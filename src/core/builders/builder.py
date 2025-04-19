"""
Contains the Builder Model.
"""
from dataclasses import dataclass
from typing import TypeVar, Type
from functools import cached_property
from enum import Enum

from helpers.utils import load_config, dynamic_import
from core.builders.base_builder import BaseBuilder
from _types.test_output import TestOutput


CONFIGS = load_config("irpd.json")


T = TypeVar("T", bound=BaseBuilder)


@dataclass(frozen=True)
class BuilderClassContainer:
    module: str
    test_class: str
    stage_name: str
    
    @cached_property
    def impl(self) -> Type[T]:
        model_class: Type[T] = dynamic_import(self.module, self.test_class)
        return model_class



class Builder(Enum):
    STAGE_0 = ("core.builders.summarization", "SummaryCSV", "0")
    STAGE_1 = ("core.builders.categorization", "CategoryPDF", "1")
    STAGE_1r = ("core.builders.categorization", "CategoryPDF", "1r")
    STAGE_1c = ("core.builders.categorization", "CategoryPDF", "1r")
    STAGE_2 = ("core.builders.classification", "ClassificationCSV", "2")
    STAGE_3 = ("core.builders.classification", "ClassificationCSV", "3")
    
    def build(self, test_output: TestOutput):
        """
        Gets the prompt composer instance.
        """
        return self.impl(test_output).build(self.stage_name)

