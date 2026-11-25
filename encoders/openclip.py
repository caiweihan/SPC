"""OpenCLIP adapter. Run this module to test image/text inference."""

from .base import METHODS_DIR, ModelAdapter, local_path, test_main

METHOD_DIR = METHODS_DIR / "OpenCLIP"


def load_model(task="image"):
    import open_clip
    import torch

    checkpoint = local_path(METHOD_DIR / "checkpoints" / "CLIP-ViT-bigG-14-laion2B-39B-b160k")
    name = f"local-dir:{checkpoint}"
    model, _, preprocess = open_clip.create_model_and_transforms(
        name, precision="fp16", device="cuda",
    )
    tokenizer = open_clip.get_tokenizer(name) if task != "image" else None
    return ModelAdapter(model.eval(), preprocess, model.visual.output_dim,
                        tokenizer=tokenizer, autocast_dtype=torch.float16)


def main():
    test_main("OpenCLIP", load_model)


if __name__ == "__main__":
    main()
