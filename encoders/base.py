"""Shared model adapter and utilities for running an encoder."""

import argparse
from contextlib import contextmanager, nullcontext
import io
import os
from pathlib import Path
import random

PROJECT_DIR = Path(__file__).resolve().parents[1]
METHODS_DIR = Path(
    os.environ.get("SPC_METHODS_DIR") or os.environ.get("AIDETECTION_METHODS_DIR") or PROJECT_DIR / "methods"
).expanduser().resolve()


def local_path(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint/resource does not exist: {path}")
    return str(path)


@contextmanager
def method_working_directory(path):
    """Scope legacy third-party resource lookup without changing CLI path semantics."""
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run in the method's environment on the GPU host.")


def to_cuda(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.cuda(non_blocking=True)
    if isinstance(value, dict):
        return {key: to_cuda(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cuda(item) for item in value)
    return value


class ModelAdapter:
    """Methods return raw image/text vectors; extraction scripts control L2 norm.

    Subclasses only override input handling and model-specific encoding.
    DataLoader transforms/collators must not capture this GPU-owning adapter.
    """

    supports_text = True

    def __init__(self, model, preprocess, feature_dim, tokenizer=None,
                 autocast_dtype=None, collate_fn=None, sigmoid_logits=False):
        self.model = model
        self.preprocess = preprocess
        self.feature_dim = int(feature_dim)
        self.tokenizer = tokenizer
        self.autocast_dtype = autocast_dtype
        self.collate_fn = collate_fn
        self.sigmoid_logits = sigmoid_logits

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def inference_context(self):
        import torch

        if self.autocast_dtype is not None:
            return torch.autocast("cuda", dtype=self.autocast_dtype)
        return nullcontext()

    def make_transform(self, jpeg_quality=None, gaussian_sigma=None):
        return ImageTransform(self.preprocess, jpeg_quality, gaussian_sigma)

    def encode_image(self, batch):
        return self.model.encode_image(to_cuda(batch))

    def encode_text(self, texts):
        if not self.supports_text or self.tokenizer is None:
            raise ValueError("This adapter has no text encoder/tokenizer loaded.")
        return self.model.encode_text(to_cuda(self.tokenizer(texts)))

    def predict_features(self, image_features, texts):
        """Image/text smoke test, including each model's learned logit scale."""
        import torch
        import torch.nn.functional as F

        image_features = F.normalize(image_features.float(), dim=-1)
        text_features = F.normalize(self.encode_text(texts).float(), dim=-1)
        logits = self.model.logit_scale.exp().float() * (image_features @ text_features.T)
        bias = getattr(self.model, "logit_bias", None)
        if bias is not None:
            logits = logits + bias.float()
        return torch.sigmoid(logits) if self.sigmoid_logits else logits.softmax(dim=-1)


class ImageTransform:
    def __init__(self, preprocess, jpeg_quality=None, gaussian_sigma=None):
        self.preprocess = preprocess
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

    def __call__(self, image):
        from PIL import Image, ImageFilter

        image = image.convert("RGB")
        if self.jpeg_quality is not None:
            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG", quality=random.randint(*self.jpeg_quality))
                buffer.seek(0)
                with Image.open(buffer) as compressed:
                    image = compressed.convert("RGB")
        elif self.gaussian_sigma is not None:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(*self.gaussian_sigma)))
        return self.preprocess(image)


class SafeCollate:
    def __init__(self, collate_fn=None):
        self.collate_fn = collate_fn

    def __call__(self, batch):
        from torch.utils.data.dataloader import default_collate

        good = [item for item in batch if item is not None]
        if not good:
            return None, None
        return (self.collate_fn or default_collate)(good)


def test_main(method, loader):
    """Shared CLI for each encoder's inference test."""
    parser = argparse.ArgumentParser(description=f"Test the local {method} image encoder.")
    parser.set_defaults(method=method)
    parser.add_argument("--image", type=Path, default=PROJECT_DIR / "datasets" / "cat.jpeg")
    parser.add_argument("--labels", nargs="+", default=["a cat", "a dog", "an animal"])
    parser.add_argument("--text-template", default="This is a photo of {}.")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")

    import torch
    from PIL import Image

    require_cuda()
    with loader("test") as adapter:
        with Image.open(args.image) as image:
            sample = adapter.preprocess(image.convert("RGB"))
        batch, _ = SafeCollate(adapter.collate_fn)([(sample, 0)])
        with torch.inference_mode(), adapter.inference_context():
            features = adapter.encode_image(batch).float()
            if features.shape != (1, adapter.feature_dim) or not torch.isfinite(features).all():
                raise RuntimeError(f"Invalid image features: shape={tuple(features.shape)}")
            print(f"Image features: {tuple(features.shape)}, L2 norm={features.norm().item():.6f}")
            if adapter.supports_text:
                texts = [args.text_template.format(label) for label in args.labels]
                probabilities = adapter.predict_features(features, texts).float()
                if not torch.isfinite(probabilities).all():
                    raise RuntimeError("Nonfinite inference probabilities")
                for label, probability in zip(args.labels, probabilities[0].tolist()):
                    print(f"{label}: {probability:.6f}")
        print("INFERENCE_OK")
