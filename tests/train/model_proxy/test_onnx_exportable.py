import pytest

from mlinstruct.train.model_proxy import OnnxExportable


def test_cannot_instantiate_onnx_exportable_directly():
    with pytest.raises(TypeError):
        OnnxExportable()


def test_torch_model_proxy_is_onnx_exportable(proxy):
    assert isinstance(proxy, OnnxExportable)
