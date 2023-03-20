from enum import Enum
from typing import Optional

from pydantic import BaseSettings


_ENV_PREFIX_OPENAI_SETTINGS = "MLSERVER_MODEL_OPENAI_"


class OpenAIModelTypeEnum(str, Enum):
    chat = "chat.completions"
    completions = "completions"
    embeddings = "embeddings"


class OpenAISettings(BaseSettings):
    """
    OpenAI settings
    """

    class Config:
        env_prefix = _ENV_PREFIX_OPENAI_SETTINGS

    api_key: str
    model_id: str
    model_type: OpenAIModelTypeEnum
    organization: Optional[str]
    llm_parameters: Optional[dict]
