"""
Smoke tests for base runtime
"""
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

from mlserver import ModelSettings
from mlserver.codecs import StringCodec
from mlserver.types import (
    ResponseOutput,
    InferenceRequest,
    InferenceResponse,
    RequestInput,
    Parameters,
)
from mlserver_llm.common import PROMPT_TEMPLATE_RESULT_FIELD
from mlserver_llm.openai.openai_runtime import OpenAIRuntime
from mlserver_llm.prompt.string_based import SimplePromptTemplate
from mlserver_llm.runtime import (
    LLMProviderRuntimeBase,
    _get_predict_parameters,
    LLMRuntime,
    _decode_and_apply_prompt,
)


@pytest.fixture
def input_values() -> dict:
    return {"i": ["asd", "qwe"]}


@pytest.fixture
def inference_request(input_values: dict) -> InferenceRequest:
    return InferenceRequest(
        inputs=[
            StringCodec.encode_input("foo", input_values["i"]),
            StringCodec.encode_input("bar", input_values["i"]),
        ]
    )


@pytest.mark.parametrize("extra", [None, {"with_prompt_template": True}])
async def test_runtime_base__smoke(inference_request: InferenceRequest, extra: dict):
    class _DummyModel(LLMProviderRuntimeBase):
        def __init__(self, settings: ModelSettings):
            super().__init__(settings)

        async def _call_impl(
            self, input_data: Any, params: Optional[dict]
        ) -> ResponseOutput:
            assert isinstance(input_data, pd.DataFrame)
            return ResponseOutput(
                name="foo", datatype="INT32", shape=[1, 1, 1], data=[1]
            )

    ml = _DummyModel(
        settings=ModelSettings(
            implementation=LLMProviderRuntimeBase, parameters={"extra": extra}
        )
    )  # dummy

    await ml.load()
    res = await ml.predict(inference_request)
    assert isinstance(res, InferenceResponse)


@pytest.mark.parametrize(
    "settings",
    [
        ModelSettings(
            implementation=LLMRuntime,
            parameters={
                "extra": {
                    "provider_id": "openai",
                    "config": {
                        "model_id": "gpt-3.5-turbo",
                        "api_key": "dummy",
                        "model_type": "chat.completions",
                    },
                }
            },
        ),
        ModelSettings(
            implementation=LLMRuntime,
            parameters={
                "extra": {
                    "provider_id": "openai",
                    "with_prompt_template": True,
                    "config": {
                        "model_id": "gpt-3.5-turbo",
                        "api_key": "dummy",
                        "model_type": "chat.completions",
                    },
                }
            },
        ),
    ],
)
async def test_runtime_factory__smoke(settings: ModelSettings):
    ml = LLMRuntime(settings=settings)
    assert isinstance(ml._rt, OpenAIRuntime)


@pytest.mark.parametrize(
    "inference_request, expected_dict",
    [
        (
            InferenceRequest(
                model_name="my-model",
                inputs=[
                    RequestInput(name="foo", datatype="INT32", shape=[1], data=[1]),
                ],
            ),
            {},
        ),
        (
            InferenceRequest(
                model_name="my-model",
                inputs=[
                    RequestInput(name="foo", datatype="INT32", shape=[1], data=[1]),
                ],
                parameters=Parameters(
                    llm_parameters={"threshold": 10, "temperature": 20}
                ),
            ),
            {"threshold": 10, "temperature": 20},
        ),
    ],
)
def test_get_llm_parameters_from_request(
    inference_request: InferenceRequest, expected_dict: dict
):
    params = _get_predict_parameters(inference_request)
    assert params == expected_dict


def test_tensor_dict_mapping(inference_request: InferenceRequest):
    prompt = SimplePromptTemplate()
    result = _decode_and_apply_prompt(prompt, inference_request)
    for col in result.columns:
        assert col == PROMPT_TEMPLATE_RESULT_FIELD
        assert result[col].values.tolist() == [
            '{"foo": ["asd", "qwe"], "bar": ["asd", "qwe"]}'
        ]
