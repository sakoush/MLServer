import abc
from abc import ABC
from typing import Any


class PromptTemplate(ABC):
    @abc.abstractmethod
    def format(self, **kwargs: Any) -> str:
        """format according to the parameters passed"""
