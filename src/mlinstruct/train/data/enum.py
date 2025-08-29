from enum import Enum

class ModelFormat(Enum):
    PT = "pt"
    ONNX = "onnx"
    SAFETENSORS = "safetensors"
    HDF5 = "h5"