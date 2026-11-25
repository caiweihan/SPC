r"""Merge H5 feature files. Run the following commands from the project root.

# Merge four training shards.
python scripts/merge_h5.py \
  outputs/PE/train-features/select/SD14-train_0.h5 \
  outputs/PE/train-features/select/SD14-train_1.h5 \
  outputs/PE/train-features/select/SD14-train_2.h5 \
  outputs/PE/train-features/select/SD14-train_3.h5 \
  -o outputs/PE/train-features/select/SD14-train.h5

# Pool the three WildRF subsets.
python scripts/merge_h5.py \
  outputs/PE/test-features/In-the-wild/WildRF-facebook.h5 \
  outputs/PE/test-features/In-the-wild/WildRF-reddit.h5 \
  outputs/PE/test-features/In-the-wild/WildRF-twitter.h5 \
  -o outputs/PE/WildRF.h5

# Merge all feature files in a directory.
python scripts/merge_h5.py \
  --input-dir outputs/PE/test-features/GenImage -o outputs/PE/GenImage.h5

# Sample the same number of rows from each input before merging.
python scripts/merge_h5.py \
  a.h5 b.h5 --equal-samples --seed 42 -o merged.h5
"""

import argparse
import os
import h5py
import numpy as np
import sys
import tempfile
from typing import List

def expand_inputs(paths: List[str]) -> List[str]:
    """Expand directory inputs into their contained .h5 files (sorted)."""
    expanded: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            h5_files = sorted(
                os.path.join(p, f) for f in os.listdir(p) if f.endswith(".h5")
            )
            if not h5_files:
                raise ValueError(f"No .h5 files found in directory: {p}")
            expanded.extend(h5_files)
        else:
            expanded.append(p)
    return expanded

def collect_class_info(paths: List[str]):
    classes = set()
    feat_dim = None
    counts = []
    for p in paths:
        with h5py.File(p, "r") as f:
            count = 0
            for name in f.keys():
                dset = f[name]
                if dset.ndim != 2:
                    raise ValueError(f"Dataset '{name}' in {p} is not 2D")
                d = dset.shape[1]
                if feat_dim is None:
                    feat_dim = d
                elif feat_dim != d:
                    raise ValueError(f"Feature-dim mismatch: {p}:{name} has dim {d}, expected {feat_dim}")
                classes.add(name)
                count += len(dset)
            counts.append(count)
    if feat_dim is None:
        raise ValueError("No input features found")
    return sorted(classes), feat_dim, counts

def merge_h5(inputs: List[str], output: str, *, equal_samples=False, seed=None):
    """Merge all rows or randomly sample the same total number from each input."""
    classes, feat_dim, counts = collect_class_info(inputs)
    n_select = min(counts) if equal_samples else None
    rng = np.random.default_rng(seed) if equal_samples else None
    with h5py.File(output, "w") as out:
        dsets = {c: out.create_dataset(name=c, shape=(0, feat_dim), maxshape=(None, feat_dim), dtype="float32") for c in classes}
        for path, count in zip(inputs, counts):
            with h5py.File(path, "r") as f:
                selected = np.sort(rng.choice(count, n_select, replace=False)) if equal_samples else None
                start = 0
                for name, source in f.items():
                    if selected is None:
                        data = source[()]
                    else:
                        end = start + len(source)
                        indices = selected[(selected >= start) & (selected < end)] - start
                        start = end
                        if not len(indices):
                            continue
                        data = source[indices]
                    if data.size == 0:
                        continue
                    arr = np.asarray(data, dtype="float32")
                    ds = dsets[name]
                    n = ds.shape[0]
                    m = arr.shape[0]
                    ds.resize((n + m, feat_dim))
                    ds[n : n + m] = arr
    return n_select

def main():
    parser = argparse.ArgumentParser(description="Merge multiple feature .h5 files (class-name datasets at root) into one.")
    parser.add_argument("inputs", nargs="*", help="Input .h5 files or directories of .h5 files to merge (order preserved)")
    parser.add_argument("--input-dir", help="Merge the .h5 files directly inside this directory (sorted)")
    parser.add_argument("--equal-samples", action="store_true",
                        help="Sample without replacement from each input file, matching the smallest file's total row count across classes")
    parser.add_argument("--seed", type=int, help="Random seed for --equal-samples")
    parser.add_argument("--delete-inputs", action="store_true", help="Delete input .h5 files after a successful merge")
    parser.add_argument("-o", "--output", required=True, help="Output .h5 file path")
    args = parser.parse_args()
    if bool(args.inputs) == (args.input_dir is not None):
        parser.error("Specify either positional inputs or --input-dir")
    if args.input_dir is not None and not os.path.isdir(args.input_dir):
        parser.error("--input-dir must be an existing directory")
    if args.seed is not None and not args.equal_samples:
        parser.error("--seed requires --equal-samples")

    try:
        inputs = expand_inputs([args.input_dir] if args.input_dir is not None else args.inputs)
        output_path = os.path.realpath(args.output)
        if args.input_dir is not None:
            inputs = [p for p in inputs if os.path.realpath(p) != output_path]
        if not inputs:
            raise ValueError("No input .h5 files to merge")
        if any(os.path.realpath(p) == output_path for p in inputs):
            raise ValueError("Output path must differ from input files")
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(os.path.abspath(args.output)),
                                         prefix=".merge-", suffix=".tmp", delete=False) as temporary:
            temporary_path = temporary.name
        try:
            n_select = merge_h5(inputs, temporary_path, equal_samples=args.equal_samples, seed=args.seed)
            os.replace(temporary_path, args.output)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        if args.delete_inputs:
            for path in dict.fromkeys(os.path.abspath(p) for p in inputs):
                os.remove(path)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(2)
    if n_select is not None:
        print(f"Merged {n_select} samples per file from {len(inputs)} files -> {args.output}")
    else:
        print("Merged", len(inputs), "files ->", args.output)

if __name__ == "__main__":
    main()
