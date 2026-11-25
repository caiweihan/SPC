r"""Extract fake/real text embeddings through the selected encoder.

Run from the project root.

CUDA_VISIBLE_DEVICES=0 python scripts/extract_text_embeddings.py \
  --method PE --fake-text "AI generated" --real-text "a real photo"

The default output is outputs/<method>/text_embeddings.npz. The NPZ stores
e_f, e_r, norm_scale, fake_text and real_text. Both vectors have L2 norm 16
by default. Existing outputs are rejected.

Use --prompt-pairs prompts.json to extract multiple pairs with one model load.
Each JSON entry specifies its own output path.

[
  {"fake_text": "AI generated", "real_text": "a real photo", "output": "outputs/PE/prompt_ablation/default.npz"}
]
"""

import argparse
import json
import math
from pathlib import Path

from extract_common import (
    PROJECT_DIR, TEXT_METHODS, add_method_argument, atomic_output, load_adapter,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_method_argument(parser)
    parser.add_argument("--fake-text", default="AI generated")
    parser.add_argument("--real-text", default="a real photo")
    parser.add_argument("--output", type=Path, help="Default: <project>/outputs/<method>/text_embeddings.npz")
    parser.add_argument("--prompt-pairs", type=Path,
                        help="JSON list of {fake_text, real_text, output}; load the model once")
    parser.add_argument("--norm-scale", type=float, default=16.0, help="L2 norm of saved text vectors (default: 16)")
    parser.set_defaults(task="text")
    return parser


def text_jobs(args):
    if args.prompt_pairs is None:
        jobs = [dict(fake_text=args.fake_text, real_text=args.real_text,
                     output=str(args.output or PROJECT_DIR / "outputs" / args.method / "text_embeddings.npz"))]
    else:
        jobs = json.loads(args.prompt_pairs.read_text())
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("--prompt-pairs must contain a nonempty JSON list")
    outputs = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or any(not isinstance(job.get(key), str) or not job[key].strip()
                                           for key in ("fake_text", "real_text", "output")):
            raise ValueError(f"Text job {index} must contain nonempty fake_text, real_text, and output strings")
        job["output"] = Path(job["output"]).expanduser().resolve()
        if job["output"] in outputs:
            raise ValueError(f"Duplicate output in text jobs: {job['output']}")
        outputs.add(job["output"])
        if job["output"].exists():
            raise FileExistsError(f"Output already exists: {job['output']}")
    return jobs


def extract_text(adapter, jobs, args):
    import numpy as np
    import torch

    with torch.inference_mode(), adapter.inference_context():
        for job in jobs:
            texts = [job["fake_text"], job["real_text"]]
            features = adapter.encode_text(texts).float()
            if features.shape != (2, adapter.feature_dim):
                raise ValueError(f"Unexpected text feature shape: {tuple(features.shape)}")
            norms = features.norm(dim=-1, keepdim=True)
            if not torch.isfinite(features).all() or (norms == 0).any():
                raise ValueError("Text encoder returned nonfinite or zero vectors")
            e_f, e_r = (args.norm_scale * features / norms.clamp_min(1e-12)).cpu().numpy()
            with atomic_output(job["output"]) as temporary:
                with temporary.open("wb") as output:
                    np.savez(output, e_f=e_f, e_r=e_r, norm_scale=np.float32(args.norm_scale),
                             fake_text=np.str_(texts[0]), real_text=np.str_(texts[1]))
            print(f"Saved: {job['output']} (e_f {e_f.shape}, e_r {e_r.shape})", flush=True)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.method not in TEXT_METHODS:
        parser.error(f"{args.method} has no text encoder; choose from {', '.join(TEXT_METHODS)}")
    if not math.isfinite(args.norm_scale) or args.norm_scale <= 0:
        parser.error("--norm-scale must be finite and positive")
    if args.prompt_pairs is not None and args.output is not None:
        parser.error("--output cannot be combined with --prompt-pairs; set output in each JSON job")
    try:
        jobs = text_jobs(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    with load_adapter(args) as adapter:
        extract_text(adapter, jobs, args)


if __name__ == "__main__":
    main()
