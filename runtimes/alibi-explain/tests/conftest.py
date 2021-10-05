import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import nest_asyncio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlserver import MLModel
from mlserver.handlers import DataPlane, ModelRepositoryHandlers
from mlserver.registry import MultiModelRegistry
from mlserver.repository import ModelRepository
from mlserver.rest import RESTServer
from mlserver.settings import ModelSettings, ModelParameters, Settings
from mlserver_alibi_explain.common import AlibiExplainSettings
from mlserver_alibi_explain.runtime import AlibiExplainRuntime
from mlserver_mlflow import MLflowRuntime

# allow nesting loop
nest_asyncio.apply()

TESTS_PATH = os.path.dirname(__file__)
# this path is coming from mlflow test data
TESTDATA_PATH = Path(__file__).parent.parent.parent / "mlflow" / "tests" / "testdata"


# TODO: how to make this in utils?
def pytest_collection_modifyitems(items):
    """
    Add pytest.mark.asyncio marker to every test.
    """
    for item in items:
        item.add_marker("asyncio")


@pytest.fixture
def pytorch_model_uri() -> str:
    pytorch_model_path = os.path.join(TESTDATA_PATH, "pytorch_model")
    if sys.version_info >= (3, 8):
        return os.path.join(pytorch_model_path, "3.8")

    return os.path.join(pytorch_model_path, "3.7")


@pytest.fixture
def model_settings_pytorch_fixed(pytorch_model_uri) -> ModelSettings:
    return ModelSettings(
        name="mlflow-model",
        parameters=ModelParameters(uri=pytorch_model_uri),
    )


@pytest.fixture
async def runtime_pytorch(model_settings_pytorch_fixed: ModelSettings) -> MLflowRuntime:
    model = MLflowRuntime(model_settings_pytorch_fixed)
    await model.load()

    return model


@pytest.fixture
def settings() -> Settings:
    return Settings(
        debug=True,
        host="127.0.0.1"
    )


@pytest.fixture
async def model_registry(runtime_pytorch) -> MultiModelRegistry:
    model_registry = MultiModelRegistry()
    await model_registry.load(runtime_pytorch)
    return model_registry


@pytest.fixture
def data_plane(settings: Settings, model_registry: MultiModelRegistry) -> DataPlane:
    return DataPlane(settings=settings, model_registry=model_registry)


@pytest.fixture
def model_repository(tmp_path, runtime_pytorch) -> ModelRepository:
    model_settings_path = tmp_path.joinpath("model-settings.json")
    model_settings_dict = {
        "name": runtime_pytorch.settings.name,
        "implementation": "mlserver_mlflow.MLflowRuntime",
        "parallel_workers": 0,
        "parameters": {
            "uri": runtime_pytorch.settings.parameters.uri,
        }
    }

    model_settings_path.write_text(json.dumps(model_settings_dict, indent=4))
    return ModelRepository(tmp_path)


@pytest.fixture
def model_repository_handlers(
    model_repository: ModelRepository, model_registry: MultiModelRegistry
) -> ModelRepositoryHandlers:
    return ModelRepositoryHandlers(
        repository=model_repository, model_registry=model_registry
    )


@pytest.fixture
async def rest_server(
    settings: Settings,
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    runtime_pytorch: MLflowRuntime,
) -> RESTServer:
    server = RESTServer(
        settings=settings,
        data_plane=data_plane,
        model_repository_handlers=model_repository_handlers,
    )

    await asyncio.gather(server.add_custom_handlers(runtime_pytorch))

    yield server

    await asyncio.gather(server.delete_custom_handlers(runtime_pytorch))


@pytest.fixture
def rest_app(rest_server: RESTServer) -> FastAPI:
    return rest_server._app


@pytest.fixture
def rest_client(rest_app: FastAPI) -> TestClient:
    return TestClient(rest_app)


@pytest.fixture
async def anchor_image_runtime(runtime_pytorch: MLModel) -> AlibiExplainRuntime:
    with patch("mlserver_alibi_explain.common.remote_predict") as remote_predict:
        def mock_predict(*args, **kwargs):
            try:
                loop = asyncio.get_event_loop()
                res = loop.run_until_complete(runtime_pytorch.predict(kwargs["v2_payload"]))
                return res
            except Exception as ex:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                res = loop.run_until_complete(runtime_pytorch.predict(kwargs["v2_payload"]))
                return res




        remote_predict.side_effect = mock_predict

        rt = AlibiExplainRuntime(
            ModelSettings(
                parallel_workers=0,
                parameters=ModelParameters(
                    uri="./data/mnist_anchor_image",
                    extra=AlibiExplainSettings(
                        explainer_type="anchor_image",
                        infer_uri=f"dummy_call"
                    )
                )
            )
        )
        await rt.load()

        return rt


@pytest.fixture
async def integrated_gradients_runtime() -> AlibiExplainRuntime:
    rt = AlibiExplainRuntime(
        ModelSettings(
            parallel_workers=1,
            parameters=ModelParameters(
                extra=AlibiExplainSettings(
                    init_parameters={
                        "n_steps": 50,
                        "method": "gausslegendre"
                    },
                    explainer_type="integrated_gradients",
                    infer_uri="./data/tf_mnist_ig/model.h5"
                )
            )
        )
    )
    await rt.load()

    return rt
