"""PE adapter. Run this module for an image/text inference test."""

import sys

from .base import METHODS_DIR, ModelAdapter, local_path, test_main

METHOD_DIR = METHODS_DIR / "PE"


def load_model(task="image"):
    import torch

    sys.path.insert(0, str(METHOD_DIR))
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms

    name = "PE-Core-G14-448"
    checkpoint = local_path(METHOD_DIR / "checkpoints" / name / f"{name}.pt")
    model = pe.CLIP.from_config(name, pretrained=True, checkpoint_path=checkpoint).cuda().eval()
    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length) if task != "image" else None
    return ModelAdapter(model, preprocess, model.visual.output_dim,
                        tokenizer=tokenizer, autocast_dtype=torch.float16)


def main():
    test_main("PE", load_model)


if __name__ == "__main__":
    main()
