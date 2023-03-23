import os
import tempfile

import pytest

from mlserver_llm.prompt.string_based import FStringPromptTemplate, SimplePromptTemplate


@pytest.mark.parametrize(
    "template, args, expected_string",
    [
        (
            "hello",
            {},
            "hello"
        ),
        (
            """
            this is a template

            for chat
            {input_1}

            {input_2}
            end
            """
            ,
            {
                "input_1": "hello there",
                "input_2": "hi"
            },
            """
            this is a template

            for chat
            hello there

            hi
            end
            """
        ),
    ],
)
def test_string_prompt_template__smoke(
        template: str, args: dict, expected_string: str):
    with tempfile.TemporaryDirectory() as tempdir:
        # you can e.g. create a file here:
        tmp_path = os.path.join(tempdir, 'template.txt')
        with open(tmp_path, 'w') as f:
            f.write(template)

        prompt = FStringPromptTemplate(tmp_path)
        res = prompt.format(**args)
        assert res == expected_string


@pytest.mark.parametrize(
    "args, expected_string",
    [
        (
            {},
            "{}"
        ),
        (
            {
                "input_1": "hello there",
                "input_2": "hi"
            },
            """{"input_1": "hello there", "input_2": "hi"}"""
        ),
    ],
)
def test_simple_prompt_template__smoke(args: dict, expected_string: str):
    prompt = SimplePromptTemplate()
    res = prompt.format(**args)
    assert res == expected_string
