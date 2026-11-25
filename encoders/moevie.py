"""MoE-ViE adapter using the official 448px packed-image preprocessing."""

import sys

from .base import METHODS_DIR, ModelAdapter, local_path, test_main

METHOD_DIR = METHODS_DIR / "MoEViE"


class MoEViEAdapter(ModelAdapter):
    def encode_image(self, batch):
        import torch
        from open_clip import image_to_device

        batch = image_to_device(batch, "cuda", torch.float32, mean=(0.5,) * 3, std=(0.5,) * 3)
        return self.model.encode_image(batch)


def load_model(task="image"):
    sys.path.insert(0, str(METHOD_DIR / "src"))
    from open_clip import create_model_and_transforms, get_tokenizer

    name = "MoEViE-H14-448"
    checkpoint = local_path(METHOD_DIR / "checkpoints" / name / f"{name}.pt")
    model, _, preprocess = create_model_and_transforms(
        name, pretrained=checkpoint,
        force_preprocess_cfg=dict(patch_size=14, size_range=(448, 448), center_crop=True, window_size=1),
        image_mean=(0.5,) * 3, image_std=(0.5,) * 3,
    )
    model = model.float().cuda().eval()
    tokenizer = get_tokenizer(name) if task != "image" else None
    return MoEViEAdapter(model, preprocess, model.visual.output_dim,
                         tokenizer=tokenizer,
                         collate_fn=preprocess.collate_fn)


def main():
    test_main("MoEViE", load_model)


if __name__ == "__main__":
    main()
