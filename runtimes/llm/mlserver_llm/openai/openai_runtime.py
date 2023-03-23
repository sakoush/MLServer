import json
from typing import Any, Optional, Tuple

import openai
import pandas as pd

from mlserver import ModelSettings
from mlserver.codecs import StringCodec
from mlserver.types import ResponseOutput
from .common import OpenAISettings, OpenAIModelTypeEnum
from ..common import PROMPT_TEMPLATE_RESULT_FIELD
from ..runtime import LLMProviderRuntimeBase


class OpenAIRuntime(LLMProviderRuntimeBase):
    """
    Runtime for OpenAI
    """

    def __init__(self, settings: ModelSettings):
        # if we are here we are sure that settings.parameters is set,
        # just helping mypy
        assert settings.parameters is not None
        assert settings.parameters.extra is not None
        config = settings.parameters.extra["config"]  # type: ignore
        self._openai_settings = OpenAISettings(**config)  # type: ignore
        if self._openai_settings.model_type != OpenAIModelTypeEnum.images:
            assert (
                self._openai_settings.model_id
            ), f"model_id required for {self._openai_settings.model_type}"
        super().__init__(settings)

    async def _call_impl(
        self, input_data: Any, params: Optional[dict]
    ) -> ResponseOutput:
        # TODO: make use of static parameters
        # TODO: implement prompt template in all valid cases

        if self._openai_settings.model_type == OpenAIModelTypeEnum.chat:
            result = await self._call_chat_impl(input_data, params)
        elif self._openai_settings.model_type == OpenAIModelTypeEnum.completions:
            result = await self._call_completions_impl(input_data, params)
        elif self._openai_settings.model_type == OpenAIModelTypeEnum.embeddings:
            result = await self._call_embeddings_impl(input_data, params)
        elif self._openai_settings.model_type == OpenAIModelTypeEnum.edits:
            result = await self._call_instruction_impl(input_data, params)
        elif self._openai_settings.model_type == OpenAIModelTypeEnum.images:
            result = await self._call_images_generations_impl(input_data, params)
        else:
            raise TypeError(f"{self._openai_settings.model_type} not supported")

        json_str = json.dumps(result)
        return StringCodec.encode_output(payload=[json_str], name="output")

    async def _call_chat_impl(self, input_data: Any, params: Optional[dict]) -> dict:
        assert isinstance(input_data, pd.DataFrame)
        if self._with_prompt_template:
            data = _prompt_to_message(input_data)
        else:
            data = _df_to_message(input_data)
        return await openai.ChatCompletion.acreate(
            api_key=self._openai_settings.api_key,
            organization=self._openai_settings.organization,
            model=self._openai_settings.model_id,
            messages=data,
            **params,  # type: ignore
        )

    async def _call_embeddings_impl(
        self, input_data: Any, params: Optional[dict]
    ) -> dict:
        assert isinstance(input_data, pd.DataFrame)
        data = _df_to_embeddings_input(input_data)
        return await openai.Embedding.acreate(
            api_key=self._openai_settings.api_key,
            organization=self._openai_settings.organization,
            model=self._openai_settings.model_id,
            input=data,
            **params,  # type: ignore
        )

    async def _call_completions_impl(
        self, input_data: Any, params: Optional[dict]
    ) -> dict:
        assert isinstance(input_data, pd.DataFrame)
        data = _df_to_completion_prompt(input_data)
        return await openai.Completion.acreate(
            api_key=self._openai_settings.api_key,
            organization=self._openai_settings.organization,
            model=self._openai_settings.model_id,
            prompt=data,
            **params,  # type: ignore
        )

    async def _call_instruction_impl(
        self, input_data: Any, params: Optional[dict]
    ) -> dict:
        assert isinstance(input_data, pd.DataFrame)
        data, instruction = _df_to_instruction(input_data)
        return await openai.Edit.acreate(
            api_key=self._openai_settings.api_key,
            organization=self._openai_settings.organization,
            model=self._openai_settings.model_id,
            input=data,
            instruction=instruction,
            **params,  # type: ignore
        )

    async def _call_images_generations_impl(
        self, input_data: Any, params: Optional[dict]
    ) -> dict:
        # note: no model_id for this api
        assert isinstance(input_data, pd.DataFrame)
        data = _df_to_images(input_data)
        return await openai.Image.acreate(
            api_key=self._openai_settings.api_key,
            organization=self._openai_settings.organization,
            prompt=data,
            **params,  # type: ignore
        )


def _prompt_to_message(df: pd.DataFrame) -> list[dict]:
    # we only pass one the first item in the list
    assert (
        PROMPT_TEMPLATE_RESULT_FIELD in df.columns
    ), f"{PROMPT_TEMPLATE_RESULT_FIELD} field not present"
    return [
        {"role": "user", "content": val}
        for val in df[PROMPT_TEMPLATE_RESULT_FIELD].values.tolist()
    ]


def _df_to_message(df: pd.DataFrame) -> list[dict]:
    assert "role" in df.columns, "user field not present"
    assert "content" in df.columns, "content field not present"
    return df[["role", "content"]].to_dict(orient="records")


def _df_to_embeddings_input(df: pd.DataFrame) -> list[dict]:
    assert "input" in df.columns, "input field not present"
    return df[["input"]].values.flatten().tolist()


def _df_to_completion_prompt(df: pd.DataFrame) -> list[dict]:
    assert "prompt" in df.columns, "prompt field not present"
    return df[["prompt"]].values.flatten().tolist()


def _df_to_instruction(df: pd.DataFrame) -> Tuple[str, str]:
    assert "input" in df.columns, "input field not present"
    assert "instruction" in df.columns, "instruction field not present"
    return df[["input", "instruction"]].values[0].flatten().tolist()


def _df_to_images(df: pd.DataFrame) -> str:
    assert "prompt" in df.columns, "prompt field not present"
    return df[["prompt"]].values[0].flatten()[0]
