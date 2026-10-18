"""Offline regression checks; no model weights or optional engine packages required."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bench
from whisper_backends import Backend, Unsupported, validate_device


class MetricsTests(unittest.TestCase):
    def test_edit_operations(self):
        self.assertEqual(bench.edit_distance("a b c".split(), "a x c d".split()), 2)
        self.assertEqual(bench.edit_distance([], ["a", "b"]), 2)
        self.assertEqual(bench.edit_distance(["a", "b"], []), 2)

    def test_korean_unicode_and_punctuation(self):
        result = bench.score("안녕하세요, WORLD！", "안녕하세요 world", "basic")
        self.assertEqual((result["wer"], result["cer"]), (0, 0))
        self.assertEqual(bench.score("한 글", "한글", "basic")["cer"], 0)
        self.assertGreater(bench.score("A", "a", "none")["wer"], 0)

    def test_empty_and_missing_reference(self):
        self.assertIsNone(bench.score(None, "hello", "basic")["wer"])
        result = bench.score("", "hello", "basic")
        self.assertIsNone(result["wer"])
        self.assertEqual(result["word_errors"], 1)
        self.assertEqual(bench.score("", "", "basic")["wer"], 0)
        self.assertEqual(bench.score("a", "a b c", "basic")["wer"], 2)

    def test_corpus_weighting_and_repeats(self):
        def row(ref, hyp, repeat=0):
            return {"seconds": 1, "audio_seconds": 2, "peak_rss_mib": 12, "repeat": repeat,
                    **bench.score(ref, hyp, "basic")}
        result = bench.summarize([row("a", "b"), row("a b c d", "a b c d"), row("a", "b", 1)])
        self.assertEqual(result["corpus_wer"], 1 / 5)
        self.assertEqual(result["scored_samples"], 2)
        self.assertEqual(result["rtf"], 0.5)


class ConfigurationTests(unittest.TestCase):
    def test_manifest_and_config_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "a.wav").touch()
            (base / "ref.txt").write_text("정답", encoding="utf-8-sig")
            (base / "models").mkdir()
            data = {"defaults": {"model": "models", "repeats": 2},
                    "samples": [{"audio": "a.wav", "reference_file": "ref.txt"}],
                    "runs": [{"engine": "faster-whisper", "repeats": 4}]}
            path = base / "config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            cfg = bench.build_runs(bench.parser().parse_args(["--config", str(path)]))[0]
            self.assertEqual(cfg["samples"][0]["reference"], "정답")
            self.assertEqual(cfg["samples"][0]["audio"], str((base / "a.wav").resolve()))
            self.assertEqual(cfg["model"], str((base / "models").resolve()))
            self.assertEqual(cfg["repeats"], 4)
            manifest = base / "manifest.jsonl"
            manifest.write_text(json.dumps(data["samples"][0]), encoding="utf-8")
            args = bench.parser().parse_args(["--manifest", str(manifest), "--engine", "openai",
                                             "--device", "cpu", "cuda:0"])
            self.assertEqual(len(bench.build_runs(args)), 2)

    def test_bad_options_rejected(self):
        good = {**bench.DEFAULTS, "engine": "openai", "samples": [{"audio": "a.wav"}]}
        for override in ({"warmup": -1}, {"repeats": 0}, {"typo": 1}, {"timeout": float("inf")},
                         {"load_options": {"device_map": "auto"}}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                bench.validate_run({**good, **override})

    def test_unsupported_and_invalid_devices(self):
        for engine, device in [("faster-whisper", "npu"), ("openai", "gpu"), ("whispercpp", "cuda:0")]:
            with self.assertRaises(Unsupported):
                validate_device(engine, device)
        for device in ("cpu:0", "cuda.foo", "cuda:-1"):
            with self.assertRaises(ValueError):
                validate_device("openai", device)
        self.assertEqual(validate_device("openvino-genai", "gpu.1"), "gpu")


class AdapterTests(unittest.TestCase):
    def adapter(self, engine):
        obj = Backend.__new__(Backend)
        obj.engine, obj.language, obj.beam = engine, "ko", 1
        obj.decode_options, obj.torch = {}, None
        return obj

    def test_faster_whisper_consumes_lazy_segments(self):
        obj = self.adapter("faster-whisper")
        events = []
        def segments():
            for text in ("안녕", " 세계"):
                events.append(text)
                yield SimpleNamespace(text=text)
        obj.model = Mock()
        obj.model.transcribe.return_value = (segments(), None)
        self.assertEqual(obj.transcribe([0]), "안녕 세계")
        self.assertEqual(len(events), 2)
        self.assertFalse(obj.model.transcribe.call_args.kwargs["vad_filter"])

    def test_hf_direct_keeps_long_audio_and_attention_mask(self):
        obj = self.adapter("hf-direct")
        obj.device, obj.dtype = "cpu", "float32"
        obj.torch = SimpleNamespace(inference_mode=nullcontext)
        tensor = Mock()
        tensor.to.return_value = tensor
        class Inputs(dict):
            def to(self, device):
                return self
        obj.processor = Mock(return_value=Inputs(input_features=tensor, attention_mask="mask"))
        obj.processor.batch_decode.return_value = ["전체 전사"]
        obj.model = Mock()
        self.assertEqual(obj.transcribe([0] * 500001), "전체 전사")
        self.assertFalse(obj.processor.call_args.kwargs["truncation"])
        self.assertTrue(obj.processor.call_args.kwargs["return_attention_mask"])
        self.assertTrue(obj.model.generate.call_args.kwargs["return_timestamps"])
        self.assertEqual(obj.model.generate.call_args.kwargs["attention_mask"], "mask")

    def test_cpp_explicit_cpu(self):
        model = Mock()
        module = SimpleNamespace(Model=model)
        cfg = {**bench.DEFAULTS, "engine": "whispercpp"}
        with patch.dict(sys.modules, {"pywhispercpp": SimpleNamespace(), "pywhispercpp.model": module}):
            Backend(cfg)
        self.assertEqual(model.call_args.kwargs["context_params"], {"use_gpu": False})

    def test_genai_text_and_language(self):
        obj = self.adapter("openvino-genai")
        obj.model = Mock()
        obj.model.generate.return_value = SimpleNamespace(texts=["안녕"])
        audio = Mock()
        audio.tolist.return_value = [0.0]
        self.assertEqual(obj.transcribe(audio), "안녕")
        self.assertEqual(obj.model.generate.call_args.kwargs["language"], "<|ko|>")

    def test_cpp_preserves_word_boundaries_between_stripped_segments(self):
        obj = self.adapter("whispercpp")
        obj.model = Mock()
        obj.model.transcribe.return_value = [SimpleNamespace(text="hello"), SimpleNamespace(text="world")]
        self.assertEqual(obj.transcribe([0]), "hello world")


class FakeMemory:
    peak = 32.0
    def __init__(self, *args):
        pass
    def current(self):
        return 16.0
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class WorkerTests(unittest.TestCase):
    def test_repeats_and_warmup_excluded_from_accuracy(self):
        model = Mock(model_id="base", details={})
        model.transcribe.return_value = "정답"
        model.device_memory.return_value = {}
        cfg = {**bench.DEFAULTS, "engine": "openai", "warmup": 2, "repeats": 3,
               "samples": [{"audio": "sample.mp4", "reference": "정답"}]}
        with patch.object(bench, "Backend", return_value=model), patch.object(bench, "MemorySampler", FakeMemory), \
                patch.object(bench, "load_audio", return_value=[0] * 16000):
            result = bench.worker(cfg)
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(model.transcribe.call_count, 5)
        self.assertEqual(len(result["rows"]), 3)
        self.assertEqual(result["summary"]["scored_samples"], 1)
        self.assertEqual(result["rows"][0]["audio_seconds"], 1)
        self.assertEqual(result["summary"]["corpus_wer"], 0)

    def test_native_process_boundary_and_skipped_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = {**bench.DEFAULTS, "engine": "whispercpp", "device": "npu", "samples": []}
            result = bench.run_isolated(cfg, Path(directory))
            self.assertEqual(result["status"], "skipped", result)
            self.assertIn("does not support", result["reason"])
            bench.export_results(Path(directory), [{**result, "id": "000-whispercpp"}])
            self.assertTrue((Path(directory) / "summary.csv").is_file())
            self.assertEqual(bench.read_json(Path(directory) / "results.json")["runs"][0]["status"], "skipped")

    def test_runtime_error_saved(self):
        with patch.object(bench, "MemorySampler", FakeMemory), \
                patch.object(bench, "Backend", side_effect=RuntimeError("compile failed")):
            result = bench.worker({**bench.DEFAULTS, "engine": "openai", "samples": []})
        self.assertEqual(result["status"], "error")
        self.assertIn("compile failed", result["traceback"])

    @unittest.skipUnless(shutil.which("ffmpeg") and importlib.util.find_spec("numpy"), "Needs FFmpeg and numpy")
    def test_real_ffmpeg_resamples_and_downmixes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setparams((2, 2, 44100, 44100, "NONE", "not compressed"))
                audio.writeframes(b"\0" * (44100 * 2 * 2))
            decoded = bench.load_audio(str(path), "ffmpeg")
            self.assertEqual(decoded.shape, (16000,))
            self.assertEqual(str(decoded.dtype), "float32")


if __name__ == "__main__":
    unittest.main()
