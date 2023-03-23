import abc
from abc import ABC
from pathlib import Path
from typing import Any, Union


class PromptTemplate(ABC):
    @abc.abstractmethod
    def format(self, **kwargs: Any) -> str:
        """format according to the parameters passed"""
