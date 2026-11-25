"""GPU integration checks for shared extraction, without loading large weights.

Run using an existing torch/torchvision environment:
    python -B -m unittest discover -s tests -p test_extract_pipeline.py
"""

import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extract_features
import extract_text_embeddings
from extract_common import ModelAdapter, SafeCollate, load_non_jpeg_as_jpeg96


class TinyModel(torch.nn.Module):
    def encode_image(self, images):
        return images.mean(dim=(-2, -1)) + 1


class TinyAdapter(ModelAdapter):
    def __init__(self):
        super().__init__(TinyModel().cuda(), transforms.ToTensor(), 3)

    def encode_text(self, texts):
        return torch.tensor([[len(text), 2, 3] for text in texts], dtype=torch.float32, device="cuda")


def dictionary_collate(items):
    # Reorder both samples and labels like an adapter with custom packing would.
    items = list(reversed(items))
    return {"pixels": torch.stack([item[0] for item in items])}, torch.tensor([item[1] for item in items])


class DictionaryAdapter(TinyAdapter):
    def __init__(self):
        super().__init__()
        self.collate_fn = dictionary_collate

    def encode_image(self, batch):
        return self.model.encode_image(batch["pixels"].cuda())


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "images"
        for cls, values in {"fake": [20, 40], "real": [70, 90, 110]}.items():
            directory = self.dataset / cls
            directory.mkdir(parents=True)
            for index, value in enumerate(values):
                Image.new("RGB", (16, 16), (value, value + 1, value + 2)).save(directory / f"{index}.png")
        (self.dataset / "real" / "broken.jpg").write_text("not an image")

    def run_image(self, output, extra=(), adapter=None, dataset=None):
        arguments = ["--method", "PE", "--val-dataset", str(dataset or self.dataset),
                     "--output-dir", str(output), *extra]
        with patch.object(extract_features, "load_adapter", return_value=adapter or TinyAdapter()) as loader:
            extract_features.main(arguments)
        return loader

    def read_rows(self, path):
        with h5py.File(path) as file:
            return {key: file[key][:] for key in file}

    def test_raw_and_normalized_features_and_metadata(self):
        raw, normalized = self.root / "raw.h5", self.root / "unit.h5"
        self.run_image(raw)
        self.run_image(normalized, ["--normalize"])
        before, after = self.read_rows(raw), self.read_rows(normalized)
        self.assertEqual(before["fake"].shape, (2, 3))
        self.assertEqual(before["real"].shape, (3, 3))
        for name in before:
            np.testing.assert_allclose(
                after[name], before[name] / np.linalg.norm(before[name], axis=1, keepdims=True),
                rtol=1e-6, atol=1e-7,
            )
            self.assertEqual(before[name].dtype, np.float32)
        with h5py.File(normalized) as file:
            self.assertTrue(file.attrs["normalized"])
            self.assertEqual(file.attrs["num_written"], 5)
            self.assertEqual(file.attrs["num_skipped"], 1)

    def test_shards_partition_the_unsharded_result(self):
        full = self.root / "full.h5"
        self.run_image(full)
        shards = []
        for index in range(2):
            output = self.root / f"shard{index}.h5"
            self.run_image(output, ["--shard", "2", str(index)])
            shards.append(self.read_rows(output))
        for name, rows in self.read_rows(full).items():
            combined = np.concatenate([shard[name] for shard in shards])
            np.testing.assert_allclose(combined[np.argsort(combined[:, 0])], rows[np.argsort(rows[:, 0])])
        empty = self.root / "empty.h5"
        self.run_image(empty, ["--shard", "20", "19"])
        self.assertTrue(all(rows.shape == (0, 3) for rows in self.read_rows(empty).values()))

    def test_custom_collate_keeps_class_alignment(self):
        regular, custom = self.root / "regular.h5", self.root / "custom.h5"
        self.run_image(regular)
        self.run_image(custom, adapter=DictionaryAdapter())
        for name, rows in self.read_rows(regular).items():
            other = self.read_rows(custom)[name]
            np.testing.assert_allclose(other[np.argsort(other[:, 0])], rows[np.argsort(rows[:, 0])])

    def test_batch_subdirs_loads_once_with_perturbation(self):
        root = self.root / "batch"
        for name in ("one", "two"):
            shutil.copytree(self.dataset, root / name)
        for flags in (["--jpeg-quality", "95", "95"], ["--gaussian-sigma", "0", "0"]):
            output = self.root / ("jpeg" if "--jpeg-quality" in flags else "blur")
            loader = self.run_image(output, ["--batch-subdirs", *flags], dataset=root)
            loader.assert_called_once()
            self.assertEqual(sorted(p.name for p in output.glob("*.h5")), ["one.h5", "two.h5"])
            for path in output.glob("*.h5"):
                self.assertEqual(sum(len(rows) for rows in self.read_rows(path).values()), 5)

    def test_failed_extraction_does_not_publish_output(self):
        output = self.root / "failed.h5"
        adapter = TinyAdapter()
        adapter.encode_image = lambda batch: torch.full((len(batch), 3), float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.run_image(output, adapter=adapter)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob("*.partial")))

    def test_all_corrupt_images_do_not_publish_output(self):
        root = self.root / "corrupt" / "fake"
        root.mkdir(parents=True)
        (root / "bad.jpg").write_text("bad")
        output = self.root / "bad.h5"
        with self.assertRaisesRegex(RuntimeError, "No valid images"):
            self.run_image(output, dataset=root.parent)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob("*.partial")))

    def test_text_prompt_pairs_scale_and_npz_schema(self):
        paths = [self.root / "a.npz", self.root / "b.npz"]
        jobs = [dict(fake_text="AI generated", real_text="a real photo", output=str(paths[0])),
                dict(fake_text="synthetic", real_text="natural", output=str(paths[1]))]
        plan = self.root / "pairs.json"
        plan.write_text(json.dumps(jobs))
        with patch.object(extract_text_embeddings, "load_adapter", return_value=TinyAdapter()) as loader:
            extract_text_embeddings.main(["--method", "pe", "--prompt-pairs", str(plan), "--norm-scale", "8"])
        loader.assert_called_once()
        for job, path in zip(jobs, paths):
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["fake_text"].item(), job["fake_text"])
                self.assertEqual(data["real_text"].item(), job["real_text"])
                self.assertEqual(data["norm_scale"], 8)
                for key in ("e_f", "e_r"):
                    self.assertEqual(data[key].dtype, np.float32)
                    self.assertAlmostEqual(np.linalg.norm(data[key]), 8, places=5)

    def test_invalid_requests_fail_before_loading_model(self):
        cases = [
            (extract_features, ["--method", "PE", "--val-dataset", str(self.dataset),
                                "--output-dir", str(self.root / "out.h5"), "--shard", "0", "0"]),
            (extract_features, ["--method", "PE", "--val-dataset", str(self.dataset),
                                "--output-dir", str(self.root / "out.h5"), "--shard", "2", "2"]),
            (extract_features, ["--method", "PE", "--val-dataset", str(self.dataset),
                                "--output-dir", str(self.root / "out.h5"), "--gaussian-sigma", "nan", "1"]),
            (extract_text_embeddings, ["--method", "DINOv3"]),
            (extract_text_embeddings, ["--method", "PE", "--norm-scale", "nan"]),
        ]
        for module, arguments in cases:
            with self.subTest(arguments=arguments), patch.object(module, "load_adapter") as loader:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    module.main(arguments)
                self.assertEqual(error.exception.code, 2)
                loader.assert_not_called()

    def test_existing_outputs_are_protected(self):
        output = self.root / "existing.h5"
        output.write_bytes(b"original")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_image(output)
        self.assertEqual(output.read_bytes(), b"original")

    def test_empty_collate_and_nested_inputs(self):
        self.assertEqual(SafeCollate()([None, None]), (None, None))
        from extract_common import to_cuda
        inputs = {"pixels": [torch.ones(2), (torch.zeros(1), "metadata")]}
        moved = to_cuda(inputs)
        self.assertEqual(moved["pixels"][1][1], "metadata")
        self.assertEqual(moved["pixels"][0].device.type, "cuda")

    def test_dinov3_skips_bad_images_without_repeating_samples(self):
        output = self.root / "dino.h5"
        self.run_image(output, ["--method", "DINOv3"])
        rows = self.read_rows(output)
        self.assertEqual((len(rows["fake"]), len(rows["real"])), (2, 3))
        with h5py.File(output) as file:
            self.assertEqual(file.attrs["num_skipped"], 1)

    def test_jpeg96_loader_checks_encoding_and_does_not_recompress_jpeg(self):
        source = Image.fromarray(np.random.default_rng(42).integers(0, 256, (32, 32, 3), dtype=np.uint8))
        png, jpeg, expected = self.root / "png.jpg", self.root / "jpeg.png", self.root / "expected.jpg"
        source.save(png, format="PNG")
        source.save(jpeg, format="JPEG", quality=85)
        source.save(expected, format="JPEG", quality=96)
        original_bytes = {path: path.read_bytes() for path in (png, jpeg)}
        with Image.open(expected) as image:
            np.testing.assert_array_equal(np.asarray(load_non_jpeg_as_jpeg96(png)), np.asarray(image.convert("RGB")))
        with Image.open(jpeg) as image:
            pixels = np.asarray(image.convert("RGB"))
        with patch.object(Image.Image, "save", side_effect=AssertionError("JPEG must not be re-encoded")):
            np.testing.assert_array_equal(np.asarray(load_non_jpeg_as_jpeg96(jpeg)), pixels)
        for path, contents in original_bytes.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_conditional_jpeg96_in_extraction_with_misleading_extensions(self):
        root = self.root / "mixed"
        for cls in ("fake", "real"):
            (root / cls).mkdir(parents=True)
        source = Image.fromarray(np.random.default_rng(7).integers(0, 256, (32, 32, 3), dtype=np.uint8))
        source.save(root / "fake/a.jpg", format="PNG")
        source.save(root / "fake/b.png", format="JPEG", quality=85)
        source.save(root / "real/a.png", format="PNG")
        reference = self.root / "reference.jpg"
        source.save(reference, format="JPEG", quality=96)
        with Image.open(reference) as image:
            compressed = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) / 255 + 1
        with Image.open(root / "fake/b.png") as image:
            original_jpeg = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) / 255 + 1
        for method in ("PE", "DINOv3"):
            with self.subTest(method=method):
                output = self.root / f"{method}-jpeg96.h5"
                self.run_image(output, ["--method", method, "--non-jpeg-to-jpeg96"], dataset=root)
                rows = self.read_rows(output)
                np.testing.assert_allclose(rows["fake"], [compressed, original_jpeg], rtol=1e-6)
                np.testing.assert_allclose(rows["real"], [compressed], rtol=1e-6)
                with h5py.File(output) as file:
                    self.assertTrue(file.attrs["non_jpeg_to_jpeg96"])
        output = self.root / "uncompressed.h5"
        self.run_image(output, dataset=root)
        uncompressed = np.asarray(source, dtype=np.float32).mean(axis=(0, 1)) / 255 + 1
        np.testing.assert_allclose(self.read_rows(output)["fake"], [uncompressed, original_jpeg], rtol=1e-6)
        with h5py.File(output) as file:
            self.assertFalse(file.attrs["non_jpeg_to_jpeg96"])

    def test_removed_cli_options_are_rejected(self):
        base = ["--method", "PE", "--val-dataset", str(self.dataset),
                "--output-dir", str(self.root / "out.h5")]
        removed = [["--no-normalize"], ["--num-shards", "2"], ["--shard-idx", "0"],
                   ["--num-workers", "1"], ["--batch-size", "1"], ["--overwrite"],
                   ["--device", "cpu"], ["--checkpoint", "weights.pt"],
                   ["--model-name", "other"], ["--config-file", "other.yaml"],
                   ["--tokenizer-path", "other"], ["--runtime-dir", "other"],
                   ["--probe-checkpoint", "other"]]
        for options in removed:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    extract_features.build_parser().parse_args(base + options)

    def test_plain_dataset_path_only(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_image(self.root / "old-format.h5", dataset=f"MyImageFolder:root={self.dataset}")

    def test_output_created_during_write_is_never_overwritten(self):
        from extract_common import atomic_output

        output = self.root / "exclusive.npz"
        with self.assertRaises(FileExistsError):
            with atomic_output(output) as temporary:
                temporary.write_bytes(b"new")
                output.write_bytes(b"existing")
        self.assertEqual(output.read_bytes(), b"existing")

    def test_existing_text_output_fails_before_loading_model(self):
        output = self.root / "existing.npz"
        output.write_bytes(b"existing")
        with patch.object(extract_text_embeddings, "load_adapter") as loader:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                extract_text_embeddings.main(["--method", "PE", "--output", str(output)])
            loader.assert_not_called()
        self.assertEqual(output.read_bytes(), b"existing")

    def test_safe_dataset_handles_missing_path_metadata(self):
        from extract_common import SafeDataset

        class BrokenDataset:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise OSError("unreadable")

        self.assertIsNone(SafeDataset(BrokenDataset())[0])


if __name__ == "__main__":
    unittest.main()
