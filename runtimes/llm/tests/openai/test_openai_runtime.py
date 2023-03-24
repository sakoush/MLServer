import json
from unittest.mock import patch

import openai
import pandas as pd
import pytest

from mlserver import ModelSettings
from mlserver.types import InferenceRequest, RequestInput, InferenceResponse
from mlserver_alibi_explain.common import convert_from_bytes
from mlserver_llm.common import PROMPT_TEMPLATE_RESULT_FIELD
from mlserver_llm.openai.openai_runtime import (
    OpenAIRuntime,
    _df_to_message,
    _df_to_embeddings_input,
    _df_to_completion_prompt,
    _df_to_instruction,
    _df_to_images,
    _prompt_to_message,
    _prompt_to_completion_prompt,
)


def _get_chat_result() -> dict:
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-0301",
        "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        "choices": [
            {
                "message": {"role": "assistant", "content": "\n\nThis is a test!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }


def _get_embeddings_result() -> dict:
    return {
        "data": [
            {
                "embedding": [
                    -0.006929283495992422,
                    -0.005336422007530928,
                    -4.547132266452536e-05,
                    -0.024047505110502243,
                ],
                "index": 0,
                "object": "embedding",
            },
            {
                "embedding": [
                    -0.006929283495992422,
                    -0.005336422007530928,
                    -4.547132266452536e-05,
                    -0.024047505110502243,
                ],
                "index": 1,
                "object": "embedding",
            },
        ],
        "model": "text-embedding-ada-002",
        "object": "list",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


def _get_completions_result() -> dict:
    return {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "text-davinci-003",
        "choices": [
            {
                "text": "\n\nThis is indeed a test",
                "index": 0,
                "logprobs": "null",
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def _get_instructions_result() -> dict:
    return {
        "object": "edit",
        "created": 1589478378,
        "choices": [
            {
                "text": "What day of the week is it?",
                "index": 0,
            }
        ],
        "usage": {"prompt_tokens": 25, "completion_tokens": 32, "total_tokens": 57},
    }


def _get_images_result() -> dict:
    return {"created": 1589478378, "data": [{"url": "https://"}, {"url": "https://"}]}


@pytest.mark.parametrize(
    "model_type, openai_interface, input_request, output_result",
    [
        (
            "chat.completions",
            "openai.ChatCompletion",
            InferenceRequest(
                inputs=[
                    RequestInput(
                        name="role", shape=[1, 1], datatype="BYTES", data=["dummy"]
                    ),
                    RequestInput(
                        name="content", shape=[1, 1], datatype="BYTES", data=["dummy"]
                    ),
                ]
            ),
            _get_chat_result(),
        ),
        (
            "completions",
            "openai.Completion",
            InferenceRequest(
                inputs=[
                    RequestInput(
                        name="prompt", shape=[1, 1], datatype="BYTES", data=["prompt"]
                    ),
                ]
            ),
            _get_completions_result(),
        ),
        (
            "embeddings",
            "openai.Embedding",
            InferenceRequest(
                inputs=[
                    RequestInput(
                        name="input", shape=[1, 1], datatype="BYTES", data=["input"]
                    ),
                ]
            ),
            _get_embeddings_result(),
        ),
        (
            "edits",
            "openai.Edit",
            InferenceRequest(
                inputs=[
                    RequestInput(
                        name="input", shape=[1, 1], datatype="BYTES", data=["i"]
                    ),
                    RequestInput(
                        name="instruction", shape=[1, 1], datatype="BYTES", data=["ins"]
                    ),
                ]
            ),
            _get_instructions_result(),
        ),
        (
            "images.generations",
            "openai.Image",
            InferenceRequest(
                inputs=[
                    RequestInput(
                        name="prompt", shape=[1, 1], datatype="BYTES", data=["image"]
                    ),
                ]
            ),
            _get_images_result(),
        ),
    ],
)
async def test_openai_runtime__smoke(
    model_type: str,
    openai_interface: str,
    input_request: InferenceRequest,
    output_result: dict,
):
    dummy_api_key = "dummy_key"
    model_id = "dummy_model"

    model_settings = ModelSettings(
        implementation=OpenAIRuntime,
        parameters={
            "extra": {
                "config": {
                    "api_key": dummy_api_key,
                    "model_id": model_id,
                    "model_type": model_type,
                }
            }
        },
    )
    rt = OpenAIRuntime(model_settings)

    async def _mocked_impl(**kwargs):
        return output_result

    with patch(openai_interface) as mock_interface:
        mock_interface.acreate = _mocked_impl
        res = await rt.predict(input_request)
        assert isinstance(res, InferenceResponse)

        output_1 = convert_from_bytes(res.outputs[1], ty=str)
        output_1_dict = json.loads(output_1)
        assert output_1_dict == output_result
        assert res.outputs[1].name == "output_all"


@pytest.mark.parametrize(
    "df, expected_messages",
    [
        (
            pd.DataFrame.from_dict({"role": ["user"], "content": ["hello"]}),
            [{"role": "user", "content": "hello"}],
        ),
        (
            pd.DataFrame.from_dict(
                {"role": ["user1", "user2"], "content": ["hello1", "hello2"]}
            ),
            [
                {"role": "user1", "content": "hello1"},
                {"role": "user2", "content": "hello2"},
            ],
        ),
    ],
)
def test_convert_df_to_messages(df: pd.DataFrame, expected_messages: list[dict]):
    messages = _df_to_message(df)
    assert messages == expected_messages


@pytest.mark.parametrize(
    "df, expected_messages",
    [
        (
            pd.DataFrame.from_dict({PROMPT_TEMPLATE_RESULT_FIELD: ["hello"]}),
            [{"role": "user", "content": "hello"}],
        ),
        (
            pd.DataFrame.from_dict(
                {PROMPT_TEMPLATE_RESULT_FIELD: ["hello1", "hello2"]}
            ),
            [
                {"role": "user", "content": "hello1"},
                {"role": "user", "content": "hello2"},
            ],
        ),
    ],
)
def test_convert_prompt_to_messages(df: pd.DataFrame, expected_messages: list[dict]):
    messages = _prompt_to_message(df)
    assert messages == expected_messages


@pytest.mark.parametrize(
    "df, expected_input",
    [
        (
            pd.DataFrame.from_dict({"input": ["this is a test input"]}),
            ["this is a test input"],
        ),
        (
            pd.DataFrame.from_dict({"input": ["input1", "input2"]}),
            [
                "input1",
                "input2",
            ],
        ),
    ],
)
def test_convert_df_to_embeddings(df: pd.DataFrame, expected_input: list[str]):
    inputs = _df_to_embeddings_input(df)
    assert inputs == expected_input


@pytest.mark.parametrize(
    "df, expected_prompt",
    [
        (
            pd.DataFrame.from_dict({"prompt": ["this is a test prompt"]}),
            ["this is a test prompt"],
        ),
        (
            pd.DataFrame.from_dict({"prompt": ["prompt1", "prompt2"]}),
            [
                "prompt1",
                "prompt2",
            ],
        ),
    ],
)
def test_convert_df_to_prompt(df: pd.DataFrame, expected_prompt: list[str]):
    prompt = _df_to_completion_prompt(df)
    assert prompt == expected_prompt


@pytest.mark.parametrize(
    "df, expected_prompt",
    [
        (
            pd.DataFrame.from_dict(
                {PROMPT_TEMPLATE_RESULT_FIELD: ["this is a test prompt"]}
            ),
            ["this is a test prompt"],
        ),
        (
            pd.DataFrame.from_dict(
                {PROMPT_TEMPLATE_RESULT_FIELD: ["prompt1", "prompt2"]}
            ),
            [
                "prompt1",
                "prompt2",
            ],
        ),
    ],
)
def test_convert_prompt_to_completion_input(
    df: pd.DataFrame, expected_prompt: list[str]
):
    prompt = _prompt_to_completion_prompt(df)
    assert prompt == expected_prompt


@pytest.mark.parametrize(
    "df, expected_input, expected_instruction",
    [
        (
            pd.DataFrame.from_dict({"input": ["dummy"], "instruction": ["hello"]}),
            "dummy",
            "hello",
        ),
        (
            pd.DataFrame.from_dict(
                {"input": ["dummy 1", "dummy 2"], "instruction": ["hello 1", "hello 2"]}
            ),
            "dummy 1",
            "hello 1",
        ),
    ],
)
def test_convert_df_to_instruction(
    df: pd.DataFrame, expected_input: str, expected_instruction: str
):
    input_string, instruction = _df_to_instruction(df)
    assert input_string == expected_input
    assert instruction == expected_instruction


@pytest.mark.parametrize(
    "df, expected_prompt",
    [
        (
            pd.DataFrame.from_dict({"prompt": ["dummy"]}),
            "dummy",
        ),
        (
            pd.DataFrame.from_dict({"prompt": ["dummy hello1", "dummy hello2"]}),
            "dummy hello1",
        ),
    ],
)
def test_convert_df_to_images(df: pd.DataFrame, expected_prompt: str):
    prompt = _df_to_images(df)
    assert prompt == expected_prompt


@pytest.mark.parametrize(
    "api_key, organization",
    [
        ("dummy_key", None),
        ("dummy_key", "dummy_org"),
    ],
)
async def test_api_key_and_org_not_set(api_key: str, organization: str):
    model_id = "gpt-3.5-turbo"

    config = {
        "api_key": api_key,
        "model_id": model_id,
        "model_type": "chat.completions",
    }
    if organization:
        config["organization"] = organization

    model_settings = ModelSettings(
        implementation=OpenAIRuntime, parameters={"extra": {"config": config}}
    )
    _ = OpenAIRuntime(model_settings)

    # check that api_key not set globally
    assert openai.api_key is None
    assert openai.organization is None
