"""Shared CLI, dataset handling, and output utilities for feature extraction."""

import argparse
from contextlib import contextmanager
import io
import math
import os
from pathlib import Path
import sys
import tempfile

# Support python scripts/extract_features.py as well as running from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoders import METHODS, TEXT_METHODS, load_encoder
from encoders.base import (
    PROJECT_DIR, ImageTransform, ModelAdapter, SafeCollate, local_path,
    method_working_directory, require_cuda, test_main, to_cuda,
)


def load_adapter(args):
    return load_encoder(args.method, args.task)


def method_name(value):
    for name in METHODS:
        if value.lower() == name.lower():
            return name
    raise argparse.ArgumentTypeError(f"Unknown method {value!r}; choose from {', '.join(METHODS)}")


def add_method_argument(parser, method=None):
    if method is None:
        parser.add_argument("--method", required=True, type=method_name, choices=METHODS)
    else:
        parser.set_defaults(method=method)


def load_non_jpeg_as_jpeg96(path):
    """Inspect the original encoding before RGB conversion discards image.format."""
    from PIL import Image

    with Image.open(path) as image:
        is_jpeg = image.format == "JPEG"
        rgb = image.convert("RGB")
    if is_jpeg:
        return rgb
    with io.BytesIO() as buffer:
        rgb.save(buffer, format="JPEG", quality=96)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class SafeDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.dataset[index]
        except Exception as error:
            try:
                base = self.dataset
                real_index = index
                while hasattr(base, "indices"):
                    real_index = base.indices[real_index]
                    base = base.dataset
                path = base.samples[real_index][0]
            except Exception:
                real_index, path = index, "unknown path"
            print(f"[WARNING] Skipping image: idx={real_index}, path={path}: {error}", flush=True)
            return None


def validate_perturbations(parser, args):
    if args.jpeg_quality is not None:
        low, high = args.jpeg_quality
        if not 1 <= low <= high <= 100:
            parser.error("--jpeg-quality requires 1 <= MIN <= MAX <= 100")
    if args.gaussian_sigma is not None:
        low, high = args.gaussian_sigma
        if not all(math.isfinite(x) for x in (low, high)) or not 0 <= low <= high:
            parser.error("--gaussian-sigma requires finite 0 <= MIN <= MAX")


@contextmanager
def atomic_output(path):
    """Publish a completed file, and remove partial output on an exception."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        yield Path(temporary)
        os.link(temporary, path)  # Atomic publication; fails if another writer created the output.
    finally:
        Path(temporary).unlink(missing_ok=True)
