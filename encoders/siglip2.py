"""SigLIP2 adapter, including processor inputs and sigmoid test probabilities."""

from .base import METHODS_DIR, ModelAdapter, local_path, test_main, to_cuda

METHOD_DIR = METHODS_DIR / "SigLIP2"


class ImageProcessor:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        return {key: value[0] for key, value in inputs.items()}


class SigLIP2Adapter(ModelAdapter):
    def encode_image(self, batch):
        return self.model.get_image_features(**to_cuda(batch))

    def encode_text(self, texts):
        inputs = self.preprocess.processor(
            text=texts, padding="max_length", max_length=64, return_tensors="pt",
        ).to("cuda")
        return self.model.get_text_features(**inputs)


def load_model(task="image"):
    import torch
    from transformers import AutoModel, AutoProcessor

    checkpoint = local_path(METHOD_DIR / "checkpoints" / "siglip2-giant-opt-patch16-384")
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).cuda().eval()
    return SigLIP2Adapter(model, ImageProcessor(processor),
                          model.config.vision_config.hidden_size,
                          autocast_dtype=torch.float16, sigmoid_logits=True)


def main():
    test_main("SigLIP2", load_model)


if __name__ == "__main__":
    main()
