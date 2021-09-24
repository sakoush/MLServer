from typing import Any

from alibi.api.interfaces import Explanation
from alibi.explainers import AnchorImage
from alibi.saving import save_explainer, load_explainer
from pydantic import BaseSettings

from mlserver import ModelSettings
from mlserver_alibi_explain.common import AlibiExplainSettings
from mlserver_alibi_explain.runtime import AlibiExplainRuntimeBase


class AnchorImageWrapper(AlibiExplainRuntimeBase):
    def __init__(self, settings: ModelSettings):
        explainer_settings = AlibiExplainSettings(**settings.parameters.extra)
        # TODO: validate the settings are ok with this specific explainer
        super().__init__(settings, explainer_settings)

    async def load(self) -> bool:
        if self.settings.parameters.uri is None:
            init_parameters = self.alibi_explain_settings.init_parameters
            self._model = AnchorImage(
                predictor=self._infer_impl,
                **init_parameters)
        else:
            # load the model from disk
            self._model = load_explainer(self.settings.parameters.uri, predictor=self._infer_impl)

        self.ready = True
        return self.ready

    def _explain_impl(self, input_data: Any, settings: BaseSettings) -> Explanation:
        explain_parameters = settings.explain_parameters
        return self._model.explain(
            input_data,
            **explain_parameters
        )


