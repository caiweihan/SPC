"""DINOv3 adapter. The backbone's normal forward returns last-layer CLS."""

from contextlib import ExitStack
import sys
import tempfile

from .base import METHODS_DIR, PROJECT_DIR, ModelAdapter, local_path, test_main, to_cuda

METHOD_DIR = METHODS_DIR / "DINOv3"


class DINOv3Adapter(ModelAdapter):
    supports_text = False

    def encode_image(self, batch):
        return self.model(to_cuda(batch))

    def make_transform(self, jpeg_quality=None, gaussian_sigma=None):
        # Preserve the native blur's placement after resize/crop, and its kernel.
        from dinov3.data.transforms import make_classification_eval_transform

        return make_classification_eval_transform(
            jpeg_quality=tuple(jpeg_quality) if jpeg_quality is not None else None,
            gaussian_sigma=tuple(gaussian_sigma) if gaussian_sigma is not None else None,
        )

    def __exit__(self, exc_type, exc_value, traceback):
        return self.runtime.__exit__(exc_type, exc_value, traceback)


def load_model(task="image"):
    sys.path.insert(0, str(METHOD_DIR))
    from dinov3.data.transforms import make_classification_eval_transform
    from dinov3.eval.setup import ModelConfig, load_model_and_context
    from dinov3.run.init import job_context

    config = local_path(METHOD_DIR / "dinov3/configs/train/dinov3_vit7b16_pretrain.yaml")
    checkpoint = local_path(METHOD_DIR / "checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth")
    runtime_root = PROJECT_DIR / "logs/DINOv3/runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    output_dir = tempfile.mkdtemp(prefix="run-", dir=runtime_root)
    with ExitStack() as runtime:
        runtime.enter_context(job_context(output_dir=output_dir))
        model, context = load_model_and_context(
            ModelConfig(config_file=config, pretrained_weights=checkpoint), output_dir=output_dir,
        )
        adapter = DINOv3Adapter(model, make_classification_eval_transform(),
                                model.embed_dim, autocast_dtype=context["autocast_dtype"])
        adapter.runtime = runtime.pop_all()
    return adapter


def main():
    test_main("DINOv3", load_model)


if __name__ == "__main__":
    main()
