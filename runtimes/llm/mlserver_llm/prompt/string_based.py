import json
from pathlib import Path
from string import Formatter
from typing import Any, Union

from mlserver_llm.prompt.base import PromptTemplate


class FStringPromptTemplate(PromptTemplate):
    def __init__(self, file: Union[Path, str]):
        with open(file, "r") as f:
            self._str = f.read()

    def format(self, **kwargs: Any) -> str:
        return Formatter().format(self._str, **kwargs)


class SimplePromptTemplate(PromptTemplate):
    def format(self, **kwargs: Any) -> str:
        return json.dumps(kwargs)
