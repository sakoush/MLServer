import pytest
import os

from mlserver.handlers import DataPlane
from mlserver.repository import ModelRepository
from mlserver import types, Settings, ModelSettings

from .models import SumModel

TESTS_PATH = os.path.dirname(__file__)
TESTDATA_PATH = os.path.join(TESTS_PATH, "testdata")


@pytest.fixture
def sum_model_settings() -> ModelSettings:
    return ModelSettings(
        name="sum-model",
        version="1.2.3",
        platform="mlserver",
        versions=["sum-model/v1.2.3"],
        inputs=[types.MetadataTensor(datatype="FP32", name="input-0", shape=[128],)],
        outputs=[types.MetadataTensor(datatype="FP32", name="output-0", shape=[1],)],
    )


@pytest.fixture
def sum_model(sum_model_settings: ModelSettings) -> SumModel:
    return SumModel(settings=sum_model_settings)


@pytest.fixture
def metadata_server_response() -> types.MetadataServerResponse:
    payload_path = os.path.join(TESTDATA_PATH, "metadata-server-response.json")
    return types.MetadataServerResponse.parse_file(payload_path)


@pytest.fixture
def metadata_model_response() -> types.MetadataModelResponse:
    payload_path = os.path.join(TESTDATA_PATH, "metadata-model-response.json")
    return types.MetadataModelResponse.parse_file(payload_path)


@pytest.fixture
def inference_request() -> types.InferenceRequest:
    payload_path = os.path.join(TESTDATA_PATH, "inference-request.json")
    return types.InferenceRequest.parse_file(payload_path)


@pytest.fixture
def inference_response() -> types.InferenceResponse:
    payload_path = os.path.join(TESTDATA_PATH, "inference-response.json")
    return types.InferenceResponse.parse_file(payload_path)


@pytest.fixture
def model_repository(sum_model: SumModel) -> ModelRepository:
    model_repository = ModelRepository()
    model_repository.load(sum_model)
    return model_repository


@pytest.fixture
def settings() -> Settings:
    settings = Settings(debug=True)
    return settings


@pytest.fixture
def data_plane(settings: Settings, model_repository: ModelRepository) -> DataPlane:
    return DataPlane(settings=settings, model_repository=model_repository)