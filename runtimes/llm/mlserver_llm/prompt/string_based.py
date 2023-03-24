import json
from pathlib import Path
from string import Formatter
from typing import Any, Union

from mlserver_llm.prompt.base import PromptTemplate


class StringPromptTemplate(PromptTemplate):
    def __init__(self, input_str: str):
        self._str = input_str

    def format(self, **kwargs: Any) -> str:
        return Formatter().format(self._str, **kwargs)


class FilePromptTemplate(StringPromptTemplate):
    def __init__(self, file: Union[Path, str]):
        with open(file, "r") as f:
            self._str = f.read()


class SimplePromptTemplate(PromptTemplate):
    def format(self, **kwargs: Any) -> str:
        return json.dumps(kwargs)
