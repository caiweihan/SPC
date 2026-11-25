"""Load only the selected image or text encoder and its dependencies."""

from importlib import import_module

from .base import require_cuda

METHODS = ("PE", "SigLIP2", "MoEViE", "OpenCLIP", "MetaCLIP", "DINOv3")
TEXT_METHODS = tuple(name for name in METHODS if name != "DINOv3")


def load_encoder(method, task="image"):
    name = method.lower()
    if not any(name == known.lower() for known in METHODS):
        raise ValueError(f"Unknown method: {method!r}")
    require_cuda()
    return import_module(f".{name}", __name__).load_model(task)
