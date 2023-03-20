import json
from unittest.mock import patch

import openai
import pandas as pd
import pytest

from mlserver import ModelSettings
from mlserver.types import InferenceRequest, RequestInput, InferenceResponse
from mlserver_alibi_explain.common import convert_from_bytes
from mlserver_llm.openai.openai_runtime import OpenAIRuntime, _df_to_messages, \
    _df_to_embeddings_input


@pytest.fixture
def chat_result() -> dict:
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


@pytest.fixture
def embeddings_result() -> dict:
    return {
      "data": [
        {
          "embedding": [
            -0.006929283495992422,
            -0.005336422007530928,
            -4.547132266452536e-05,
            -0.024047505110502243
          ],
          "index": 0,
          "object": "embedding"
        }
      ],
      "model": "text-embedding-ada-002",
      "object": "list",
      "usage": {
        "prompt_tokens": 5,
        "total_tokens": 5
      }
    }


async def test_openai_chat__smoke(chat_result: dict):
    dummy_api_key = "dummy_key"
    model_id = "dummy_model"

    model_settings = ModelSettings(
        implementation=OpenAIRuntime,
        parameters={
            "extra": {
                "config":
                    {
                        "api_key": dummy_api_key,
                        "model_id": model_id,
                        "model_type": "chat.completions"
                    }
            }
        },
    )
    rt = OpenAIRuntime(model_settings)

    async def _mocked_chat_impl(**kwargs):
        return chat_result

    with patch("openai.ChatCompletion") as mock_chat:
        mock_chat.acreate = _mocked_chat_impl
        res = await rt.predict(
            InferenceRequest(
                inputs=[
                    RequestInput(
                        name="role", shape=[1, 1], datatype="BYTES", data=["dummy"]
                    ),
                    RequestInput(
                        name="content", shape=[1, 1], datatype="BYTES", data=["dummy"]
                    ),
                ]
            )
        )
        assert isinstance(res, InferenceResponse)
        output = convert_from_bytes(res.outputs[0], ty=str)
        output_dict = json.loads(output)
        assert output_dict == chat_result
        assert res.outputs[0].name == "output"


async def test_openai_embeddings__smoke(embeddings_result: dict):
    dummy_api_key = "dummy_key"
    model_id = "dummy_model"

    model_settings = ModelSettings(
        implementation=OpenAIRuntime,
        parameters={
            "extra": {
                "config":
                    {
                        "api_key": dummy_api_key,
                        "model_id": model_id,
                        "model_type": "embeddings"
                    }
            }
        },
    )
    rt = OpenAIRuntime(model_settings)

    async def _mocked_embeddings_impl(**kwargs):
        return embeddings_result

    with patch("openai.Embedding") as mock_embedings:
        mock_embedings.acreate = _mocked_embeddings_impl
        res = await rt.predict(
            InferenceRequest(
                inputs=[
                    RequestInput(
                        name="input", shape=[1, 1], datatype="BYTES", data=["dummy"]
                    ),
                ]
            )
        )
        assert isinstance(res, InferenceResponse)
        output = convert_from_bytes(res.outputs[0], ty=str)
        output_dict = json.loads(output)
        assert output_dict == embeddings_result
        assert res.outputs[0].name == "output"


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
    messages = _df_to_messages(df)
    assert messages == expected_messages


@pytest.mark.parametrize(
    "df, expected_input",
    [
        (
            pd.DataFrame.from_dict({"input": ["this is a test input"]}),
            ["this is a test input"],
        ),
        (
            pd.DataFrame.from_dict(
                {"input": ["input1", "input2"]}
            ),
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
        "model_type": "chat.completions"
    }
    if organization:
        config["organization"] = organization

    model_settings = ModelSettings(
        implementation=OpenAIRuntime, parameters={
            "extra": {
                "config": config
            }
        }
    )
    _ = OpenAIRuntime(model_settings)

    # check that api_key not set globally
    assert openai.api_key is None
    assert openai.organization is None
