"""MetaCLIP2 adapter. Run this module to test image/text inference."""

import sys

from .base import METHODS_DIR, ModelAdapter, method_working_directory, local_path, test_main

METHOD_DIR = METHODS_DIR / "MetaCLIP"


def load_model(task="image"):
    import torch

    sys.path.insert(0, str(METHOD_DIR))
    from src.mini_clip.factory import create_model_and_transforms, get_tokenizer

    name = "ViT-bigG-14-378-worldwide@WorldWideCLIP"
    checkpoint = local_path(METHOD_DIR / "checkpoints" / "metaclip2_bigG14_378px_worldwide.pt")
    # The upstream factory scans os.listdir('src/mini_clip') internally.
    with method_working_directory(METHOD_DIR):
        model, _, preprocess = create_model_and_transforms(name, pretrained=checkpoint)
    tokenizer = None
    if task != "image":
        tokenizer = get_tokenizer(local_path(METHOD_DIR / "checkpoints" / "xlm-v-base"))
    return ModelAdapter(model.cuda().eval(), preprocess, model.visual.output_dim,
                        tokenizer=tokenizer, autocast_dtype=torch.float16)


def main():
    test_main("MetaCLIP", load_model)


if __name__ == "__main__":
    main()
