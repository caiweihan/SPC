r"""Extract image features through the selected encoder, without changing cwd.

Run the following commands from the project root.

Input directories use ImageFolder layout, with class subdirectories such as
fake/ and real/. With --batch-subdirs, each immediate subdirectory is a
separate ImageFolder and --output-dir is a directory of H5 files.

Outputs store float32 features by class and record normalization, sharding,
perturbations and the number of skipped images. Unreadable images are skipped.
Existing outputs are rejected.
--non-jpeg-to-jpeg96 checks the decoded format and re-encodes non-JPEG images
in memory before any requested JPEG or Gaussian perturbation.

# Extract features from one dataset.
CUDA_VISIBLE_DEVICES=0 python scripts/extract_features.py --method PE \
  --val-dataset datasets/Wild-Datasets/Chameleon \
  --output-dir outputs/PE/test-features/In-the-wild/Chameleon.h5

# Extract one of four training shards. Use indices 0 through 3 and separate output files.
CUDA_VISIBLE_DEVICES=0 python scripts/extract_features.py --method PE \
  --val-dataset datasets/GenImage/sdv4/train --shard 4 0 \
  --output-dir outputs/PE/train-features/select/SD14-train_0.h5

# Apply JPEG compression at quality 60.
CUDA_VISIBLE_DEVICES=0 python scripts/extract_features.py --method PE \
  --val-dataset datasets/GenImage/sdv4/train --jpeg-quality 60 60 \
  --output-dir outputs/PE/train-features/SD14-train_jpeg.h5

# Apply Gaussian blur with sigma 0.5.
CUDA_VISIBLE_DEVICES=0 python scripts/extract_features.py --method PE \
  --val-dataset datasets/Wild-Datasets/Chameleon --gaussian-sigma 0.5 0.5 \
  --output-dir outputs/PE/Chameleon-blur05.h5
"""

import argparse
import json
from pathlib import Path

from extract_common import (
    SafeCollate, SafeDataset, add_method_argument, atomic_output,
    load_adapter, load_non_jpeg_as_jpeg96, validate_perturbations,
)

BATCH_SIZE = 4
NUM_WORKERS = 5


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_method_argument(parser)
    parser.add_argument("--val-dataset", required=True, help="ImageFolder root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output H5 file; output folder with --batch-subdirs")
    parser.add_argument("--shard", type=int, nargs=2, default=(1, 0), metavar=("NUM", "INDEX"),
                        help="Shard count and zero-based index (default: 1 0)")
    parser.add_argument("--batch-subdirs", action="store_true", help="Process immediate subdirectories; load the model once")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--jpeg-quality", type=int, nargs=2, metavar=("MIN", "MAX"))
    group.add_argument("--gaussian-sigma", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--normalize", action="store_true", help="L2-normalize image vectors (default: disabled)")
    parser.add_argument("--non-jpeg-to-jpeg96", action="store_true",
                        help="Re-encode non-JPEG images as JPEG quality 96 before preprocessing, using image.format")
    parser.set_defaults(task="image")
    return parser


def extraction_jobs(args):
    root = Path(args.val_dataset).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    output = args.output_dir.expanduser().resolve()
    if args.batch_subdirs:
        children = sorted(child for child in root.iterdir() if child.is_dir())
        if not children:
            raise ValueError(f"No dataset subdirectories found in {root}")
        if output == root or root in output.parents:
            raise ValueError("With --batch-subdirs, the output folder must be outside the dataset root")
        jobs = [(child, output / f"{child.name}.h5") for child in children]
    else:
        jobs = [(root, output)]
    for _, path in jobs:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
    return jobs


def extract_dataset(adapter, dataset_path, output_path, args):
    import h5py
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder
    from tqdm import tqdm

    transform = adapter.make_transform(args.jpeg_quality, args.gaussian_sigma)
    loader_options = {"loader": load_non_jpeg_as_jpeg96} if args.non_jpeg_to_jpeg96 else {}
    dataset = ImageFolder(str(dataset_path), transform=transform, **loader_options)
    class_names = dataset.classes
    num_shards, shard_idx = args.shard
    if num_shards > 1:
        dataset = Subset(dataset, range(shard_idx, len(dataset), num_shards))
    loader = DataLoader(
        SafeDataset(dataset), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        shuffle=False, drop_last=False, pin_memory=True,
        collate_fn=SafeCollate(adapter.collate_fn),
    )
    print(f"Dataset: {dataset_path}\nClasses: {class_names}\nSamples in shard: {len(dataset)}\n"
          f"L2 normalize: {args.normalize}\nSaving to: {output_path}", flush=True)
    if args.non_jpeg_to_jpeg96:
        print("Non-JPEG images are re-encoded as JPEG quality 96 before preprocessing.", flush=True)
    written = 0
    with atomic_output(output_path) as temporary:
        with h5py.File(temporary, "w") as output:
            dsets = {
                name: output.create_dataset(name, shape=(0, adapter.feature_dim),
                                            maxshape=(None, adapter.feature_dim), dtype="float32")
                for name in class_names
            }
            output.attrs.update(
                method=args.method, normalized=args.normalize, dataset=str(dataset_path),
                classes=json.dumps(class_names), shard_idx=shard_idx, num_shards=num_shards,
                jpeg_quality=json.dumps(args.jpeg_quality), gaussian_sigma=json.dumps(args.gaussian_sigma),
                non_jpeg_to_jpeg96=args.non_jpeg_to_jpeg96,
            )
            with torch.inference_mode(), adapter.inference_context():
                for batch, labels in tqdm(loader, desc=f"Shard {shard_idx}/{num_shards}"):
                    if batch is None:
                        continue
                    features = adapter.encode_image(batch).float()
                    if features.shape != (len(labels), adapter.feature_dim):
                        raise ValueError(f"Unexpected feature shape: {tuple(features.shape)}")
                    if not torch.isfinite(features).all():
                        raise ValueError("Nonfinite image features")
                    if args.normalize:
                        features = F.normalize(features, dim=-1)
                    features = features.cpu().numpy()
                    labels = labels.long().cpu().numpy()
                    for label in np.unique(labels):
                        rows = features[labels == label]
                        target = dsets[class_names[int(label)]]
                        start = target.shape[0]
                        target.resize(start + len(rows), axis=0)
                        target[start:] = rows
                    written += len(features)
            if len(dataset) and not written:
                raise RuntimeError(f"No valid images could be extracted from {dataset_path}")
            output.attrs.update(num_written=written, num_skipped=len(dataset) - written)
    print(f"Done: {written} images, {len(dataset) - written} skipped -> {output_path}", flush=True)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    num_shards, shard_idx = args.shard
    if num_shards < 1 or not 0 <= shard_idx < num_shards:
        parser.error("--shard requires NUM >= 1 and 0 <= INDEX < NUM")
    validate_perturbations(parser, args)
    try:
        jobs = extraction_jobs(args)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    with load_adapter(args) as adapter:
        for index, (dataset_path, output_path) in enumerate(jobs, 1):
            print(f"[{index}/{len(jobs)}]", flush=True)
            extract_dataset(adapter, dataset_path, output_path, args)


if __name__ == "__main__":
    main()
