from typing import Any, Optional

import pandas as pd

from mlserver.codecs import (
    PandasCodec,
)
from mlserver.model import MLModel
from mlserver.model_wrapper import WrapperMLModel
from mlserver.settings import ModelSettings
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    ResponseOutput,
)
from mlserver.utils import get_model_uri
from .common import LLM_CALL_PARAMETERS_TAG, PROVIDER_ID_TAG
from .dependency_reference import get_mlmodel_class_as_str, import_and_get_class
from .prompt.base import PromptTemplateBase
from .prompt.string_based import FStringPromptTemplate


class LLMProviderRuntimeBase(MLModel):
    """
    Base class for LLM models hosted by a provider (e.g. OpenAI)
    """

    def __init__(self, settings: ModelSettings):
        self._prompt_template: Optional[PromptTemplateBase] = None
        if self._settings.parameters.extra:
            self._with_prompt_template = self._settings.parameters.extra.get(
                "with_prompt_template", default=False)
        else:
            self._with_prompt_template = False
        super().__init__(settings)

    async def load(self) -> bool:
        # if uri is not none there is a prompt template to load
        if self._with_prompt_template:
            if self._settings.parameters.uri:
                prompt_template_uri = await get_model_uri(self._settings)
                self._prompt_template = FStringPromptTemplate(prompt_template_uri)

        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """
        This will call the model endpoint for inference
        """

        # TODO: what are the better codecs for the different types of openai models?
        input_data = PandasCodec.decode_request(payload)
        call_parameters = _get_predict_parameters(payload)
        # TODO: deal with error and retries
        if self._prompt_template:
            input_data = _apply_prompt_template(input_data)  # TODO

        output_data = await self._call_impl(input_data, call_parameters)

        return InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=[output_data],
        )

    async def _call_impl(
        self, input_data: Any, params: Optional[dict]
    ) -> ResponseOutput:
        raise NotImplementedError


def _get_predict_parameters(payload: InferenceRequest) -> dict:
    runtime_parameters = dict()
    if payload.parameters is not None:
        settings_dict = payload.parameters.dict()
        if LLM_CALL_PARAMETERS_TAG in settings_dict:
            runtime_parameters = settings_dict[LLM_CALL_PARAMETERS_TAG]
    return runtime_parameters


def _apply_prompt_template(df: pd.DataFrame) -> pd.DataFrame:
    # TODO
    pass


class LLMRuntime(WrapperMLModel):
    """Wrapper / Factory class for specific llm providers"""

    def __init__(self, settings: ModelSettings):
        assert settings.parameters is not None
        assert PROVIDER_ID_TAG in settings.parameters.extra  # type: ignore

        provider_id = settings.parameters.extra[PROVIDER_ID_TAG]  # type: ignore

        rt_class = import_and_get_class(get_mlmodel_class_as_str(provider_id))

        self._rt = rt_class(settings)
