from typing import Any, Optional

import pandas as pd
from numpy import ndarray

from mlserver.codecs import (
    PandasCodec,
)
from mlserver.model import MLModel
from mlserver.settings import ModelSettings
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
)
from mlserver.utils import get_model_uri
from mlserver_mlflow import TensorDictCodec
from .common import (
    LLM_CALL_PARAMETERS_TAG,
    PROVIDER_ID_TAG,
    PROMPT_TEMPLATE_RESULT_FIELD,
    PROMPT_TEMPLATE_INPUT_FIELD,
)
from .dependency_reference import get_mlmodel_class_as_str, import_and_get_class
from .prompt.base import PromptTemplate
from .prompt.string_based import (
    FilePromptTemplate,
    SimplePromptTemplate,
    StringPromptTemplate,
)


class LLMProviderRuntimeBase(MLModel):
    """
    Base class for LLM models hosted by a provider (e.g. OpenAI)
    """

    def __init__(self, settings: ModelSettings):
        self._static_prompt_template: Optional[PromptTemplate] = None
        if settings.parameters and settings.parameters.extra:
            self._with_prompt_template = settings.parameters.extra.get(
                "with_prompt_template", False
            )
        else:
            self._with_prompt_template = False
        super().__init__(settings)

    async def load(self) -> bool:
        # if uri is not none there is a prompt template to load
        if self._with_prompt_template:
            if self._settings.parameters and self._settings.parameters.uri:
                prompt_template_uri = await get_model_uri(self._settings)
                self._static_prompt_template = FilePromptTemplate(prompt_template_uri)
            else:
                self._static_prompt_template = SimplePromptTemplate()

        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """
        This will call the model endpoint for inference
        """
        call_parameters = _get_predict_parameters(payload)
        # TODO: deal with error and retries
        if self._static_prompt_template:
            input_data = _decode_and_apply_prompt(self._static_prompt_template, payload)
        else:
            input_data = self.decode_request(payload, default_codec=PandasCodec)

        output_data = await self._call_impl(input_data, call_parameters)

        return InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=output_data,
        )

    async def _call_impl(
        self, input_data: Any, params: Optional[dict]
    ) -> list[ResponseOutput]:
        raise NotImplementedError


def _get_predict_parameters(payload: InferenceRequest) -> dict:
    runtime_parameters = dict()
    if payload.parameters is not None:
        settings_dict = payload.parameters.dict()
        if LLM_CALL_PARAMETERS_TAG in settings_dict:
            runtime_parameters = settings_dict[LLM_CALL_PARAMETERS_TAG]
    return runtime_parameters


def _decode_and_apply_prompt(
    prompt: PromptTemplate, payload: InferenceRequest
) -> pd.DataFrame:
    # TODO: implement a variation of `TensorDictCodec` to produce dict[str, list[str]]
    input_data_raw = TensorDictCodec.decode_request(payload)
    input_data = _apply_prompt_template(prompt, input_data_raw)
    return input_data


def _apply_prompt_template(
    static_prompt_template: PromptTemplate, input_data: dict[str, ndarray]
) -> pd.DataFrame:
    # TODO: we need to think more about the data manipulations done here
    # refer to tests for use cases
    prompt_template = static_prompt_template
    if PROMPT_TEMPLATE_INPUT_FIELD in input_data:
        prompt_template_str = list(input_data[PROMPT_TEMPLATE_INPUT_FIELD])[0]
        if isinstance(prompt_template_str, bytes):
            prompt_template_str = prompt_template_str.decode("utf-8")
        prompt_template = StringPromptTemplate(prompt_template_str)

    data_dict = {
        k: [i.decode("utf-8") if isinstance(i, bytes) else i for i in val]
        for k, val in input_data.items()
        if k != PROMPT_TEMPLATE_INPUT_FIELD
    }
    prompt = prompt_template.format(**data_dict)
    return pd.DataFrame([prompt], columns=[PROMPT_TEMPLATE_RESULT_FIELD])


class LLMRuntime:
    """Wrapper / Factory class for specific llm providers"""

    def __new__(cls, settings: ModelSettings):
        assert settings.parameters is not None
        assert PROVIDER_ID_TAG in settings.parameters.extra  # type: ignore

        provider_id = settings.parameters.extra[PROVIDER_ID_TAG]  # type: ignore

        rt_class = import_and_get_class(get_mlmodel_class_as_str(provider_id))

        return rt_class(settings)
