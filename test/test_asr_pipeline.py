"""Offline behavioral checks: no model packages, downloads or GPU required."""
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from videosubx.asr import (Chunk, Decoded, DecodeConfig, TranscriptionPipeline,
                          OpenAIBackend, TransformersBackend, FasterWhisperBackend,
                          WhisperCppBackend, WhisperXVAD, align_result, load_audio,
                          load_backend, merge_regions)

RATE = 16000


class RecordingBackend:
    name, supports_batch = "recording", True

    def __init__(self):
        self.calls = []

    def decode(self, chunks, *, language):
        self.calls.append((list(chunks), language))
        return [Decoded(str(c.start)) for c in chunks]


class PipelineTests(unittest.TestCase):
    def test_real_slices_keep_internal_silence_and_original_time(self):
        audio = np.ones(46 * RATE, dtype=np.float32)
        audio[5 * RATE:8 * RATE] = 0
        backend = RecordingBackend()
        regions = [{"start": 2, "end": 5}, {"start": 8, "end": 12},
                   {"start": 40, "end": 45}]
        result = TranscriptionPipeline(backend).transcribe(audio, language="ko", speech_regions=regions)
        chunks, lang = backend.calls[0]
        self.assertEqual(lang, "ko")
        self.assertEqual([(s["start"], s["end"]) for s in result["segments"]], [(2, 12), (40, 45)])
        self.assertEqual([c.valid_samples for c in chunks], [10 * RATE, 5 * RATE])
        for chunk in chunks:
            self.assertEqual(chunk.audio.shape, (30 * RATE,))
            self.assertEqual(chunk.audio.dtype, np.float32)
            self.assertTrue(np.all(chunk.audio[chunk.valid_samples:] == 0))
        np.testing.assert_array_equal(chunks[0].audio[:10 * RATE], audio[2 * RATE:12 * RATE])
        self.assertTrue(np.all(chunks[0].audio[3 * RATE:6 * RATE] == 0))

    def test_batches_tail_order_and_progress(self):
        backend = RecordingBackend()
        regions = [{"start": s, "end": s + 1} for s in (0, 31, 62, 93, 124)]
        progress = []
        result = TranscriptionPipeline(backend).transcribe(np.zeros(125 * RATE, np.float32),
                    language="en", speech_regions=regions, batch_size=2, progress_callback=progress.append)
        self.assertEqual([len(c[0]) for c in backend.calls], [2, 2, 1])
        self.assertEqual([s["start"] for s in result["segments"]], [0, 31, 62, 93, 124])
        self.assertEqual(progress, [40, 80, 100])

    def test_nonbatch_engine_is_sequential_and_warns(self):
        backend = RecordingBackend()
        backend.supports_batch = False
        with self.assertWarnsRegex(UserWarning, "sequentially"):
            TranscriptionPipeline(backend).transcribe(np.zeros(35 * RATE), language="en",
                speech_regions=[{"start": 0, "end": 1}, {"start": 31, "end": 32}], batch_size=8)
        self.assertEqual([len(c[0]) for c in backend.calls], [1, 1])

    def test_merge_uses_elapsed_span_and_rejects_unsafe_regions(self):
        regions = [{"start": 0, "end": 1}, {"start": 29, "end": 30}, {"start": 31, "end": 32}]
        self.assertEqual(merge_regions(regions, 32), [{"start": 0, "end": 30}, {"start": 31, "end": 32}])
        bad = [[{"start": -1, "end": 1}], [{"start": 0, "end": 31}],
               [{"start": 5, "end": 6}, {"start": 1, "end": 2}],
               [{"start": 0, "end": 5}, {"start": 4, "end": 6}],
               [{"start": 0, "end": float("nan")}], [{"start": 40, "end": 41}]]
        for regions in bad:
            with self.subTest(regions=regions), self.assertRaises(ValueError):
                merge_regions(regions, 35)
        self.assertEqual(merge_regions([{"start": 0, "end": 1.01}], 1), [{"start": 0, "end": 1}])

    def test_small_chunk_size_still_pads_to_thirty(self):
        backend = RecordingBackend()
        TranscriptionPipeline(backend, chunk_size=15).transcribe(np.zeros(16 * RATE), language="ko",
            speech_regions=[{"start": 0, "end": 10}, {"start": 11, "end": 16}])
        self.assertEqual(len(backend.calls[0][0]), 2)
        self.assertEqual(backend.calls[0][0][1].audio.size, 30 * RATE)

    def test_silence_does_not_call_decoder_or_alignment_loader(self):
        backend = RecordingBackend()
        result = TranscriptionPipeline(backend).transcribe(np.zeros(RATE), language="ko", speech_regions=[])
        self.assertEqual(result, {"segments": [], "language": "ko"})
        self.assertEqual(backend.calls, [])
        self.assertEqual(align_result(result, np.zeros(RATE))["word_segments"], [])

    def test_custom_detector_and_precomputed_override(self):
        backend = RecordingBackend()
        vad = Mock(return_value=[{"start": 0, "end": 1}])
        pipe = TranscriptionPipeline(backend, vad)
        pipe.transcribe(np.zeros(RATE), language="ko")
        vad.assert_called_once()
        pipe.transcribe(np.zeros(RATE), language="ko", speech_regions=[])
        vad.assert_called_once()

    def test_invalid_inputs_and_backend_cardinality_fail(self):
        backend = RecordingBackend()
        pipe = TranscriptionPipeline(backend)
        for options in ({"language": "auto"}, {"language": "ko", "batch_size": 0},
                        {"language": "ko", "batch_size": True}):
            with self.assertRaises(ValueError):
                pipe.transcribe(np.zeros(RATE), speech_regions=[], **options)
        with self.assertRaises(ValueError):
            pipe.transcribe(np.zeros(RATE), language="ko")
        for audio in (np.zeros((2, RATE)), np.zeros(RATE, np.int16), np.array([np.nan])):
            with self.assertRaises(ValueError):
                pipe.transcribe(audio, language="ko", speech_regions=[])
        backend.decode = Mock(return_value=[])
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            pipe.transcribe(np.zeros(RATE), language="ko", speech_regions=[{"start": 0, "end": 1}])

    def test_ffmpeg_decodes_stereo_non16k_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(2)
                stream.setsampwidth(2)
                stream.setframerate(8000)
                stream.writeframes(np.full((8000, 2), 1000, dtype="<i2").tobytes())
            audio = load_audio(path)
            self.assertEqual(audio.shape, (RATE,))
            self.assertEqual(audio.dtype, np.float32)
            self.assertAlmostEqual(float(audio.mean()), 1000 / 32768, places=5)

    def test_alignment_uses_original_unpadded_audio(self):
        original = np.ones(RATE)
        result = {"segments": [{"start": .1, "end": .5, "text": "hi"}], "language": "en"}
        wx = SimpleNamespace(load_align_model=Mock(return_value=("model", "meta")),
                             align=Mock(return_value={"segments": [], "word_segments": []}))
        with patch.dict(sys.modules, {"whisperx": wx}):
            aligned = align_result(result, original, device="cpu")
        self.assertIs(wx.align.call_args.args[3], original)
        self.assertEqual(aligned["language"], "en")


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class AdapterTests(unittest.TestCase):
    def chunks(self):
        return [Chunk(np.zeros(30 * RATE, np.float32), RATE, 0, 1),
                Chunk(np.zeros(30 * RATE, np.float32), 2 * RATE, 31, 33)]

    def test_cpp_independent_text_only_settings(self):
        model = Mock()
        model.transcribe.side_effect = [[SimpleNamespace(text="hello")], [SimpleNamespace(text="world")]]
        adapter = WhisperCppBackend(model, config=DecodeConfig(5, "names"))
        self.assertEqual([r.text for r in adapter.decode(self.chunks(), language="en")], ["hello", "world"])
        for call in model.transcribe.call_args_list:
            self.assertEqual(call.args[0].size, 30 * RATE)
            for key, value in {"no_context": True, "no_timestamps": True,
                               "token_timestamps": False, "single_segment": True,
                               "temperature_inc": 0.0, "initial_prompt": "names"}.items():
                self.assertEqual(call.kwargs[key], value)

    def test_hf_and_optimum_use_real_lengths_mask_shortform_and_prompt(self):
        for name in ("huggingface", "optimum-intel", "optimum-onnx"):
            processor = Mock(return_value=FakeInputs(input_features="features", attention_mask="mask"))
            processor.batch_decode.return_value = ["hello", "world"]
            model = Mock()
            model.generation_config = SimpleNamespace(is_multilingual=True)
            adapter = TransformersBackend(model, processor, name=name, config=DecodeConfig(5, "names"))
            with patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)}):
                results = adapter.decode(self.chunks(), language="ko")
            self.assertEqual([r.text for r in results], ["hello", "world"])
            self.assertEqual([len(a) for a in processor.call_args.args[0]], [RATE, 2 * RATE])
            self.assertEqual(processor.call_args.kwargs["max_length"], 30 * RATE)
            self.assertTrue(processor.call_args.kwargs["return_attention_mask"])
            options = model.generate.call_args.kwargs
            self.assertEqual(options["attention_mask"], "mask")
            self.assertFalse(options["return_timestamps"])
            self.assertFalse(options["return_token_timestamps"])
            self.assertFalse(options["condition_on_prev_tokens"])
            self.assertEqual(options["prompt_condition_type"], "first-segment")

    def test_faster_whisper_no_longform_no_vad_and_exact_mel_batch(self):
        tokenizer = SimpleNamespace(eot=100, encode=Mock(return_value=[1]),
                                    decode=lambda ids: str(ids))
        model = Mock()
        model.model.is_multilingual = True
        model.max_length = 448
        model.get_prompt.return_value = [10, 20, 30]
        model.feature_extractor.side_effect = [np.zeros((80, 3000)), np.zeros((80, 3000))]
        model.model.generate.return_value = [SimpleNamespace(sequences_ids=[[1, 2, 100]], scores=[-1.0]),
                                             SimpleNamespace(sequences_ids=[[3, 100]], scores=[-.5])]
        fake_modules = {"faster_whisper.tokenizer": SimpleNamespace(Tokenizer=Mock(return_value=tokenizer)),
                        "faster_whisper.transcribe": SimpleNamespace(get_suppressed_tokens=Mock(return_value=[10]))}
        with patch.dict(sys.modules, fake_modules):
            results = FasterWhisperBackend(model).decode(self.chunks(), language="ko")
        self.assertEqual([r.text for r in results], ["[1, 2]", "[3]"])
        self.assertEqual(model.encode.call_args.args[0].shape, (2, 80, 3000))
        for call in model.feature_extractor.call_args_list:
            self.assertEqual(call.kwargs["padding"], 0)
        model.transcribe.assert_not_called()
        self.assertEqual(len(model.model.generate.call_args.args[1]), 2)
        self.assertTrue(model.get_prompt.call_args.kwargs["without_timestamps"])

    def test_openai_uses_decode_not_sequential_transcribe(self):
        tensor = Mock()
        whisper = SimpleNamespace(log_mel_spectrogram=Mock(return_value="mel"),
                                  DecodingOptions=Mock(), decode=Mock(return_value=[
                                      SimpleNamespace(text="a", avg_logprob=-1),
                                      SimpleNamespace(text="b", avg_logprob=-2)]))
        torch = SimpleNamespace(inference_mode=nullcontext, stack=Mock(return_value=tensor))
        model = Mock()
        model.dims.n_mels = 128
        with patch.dict(sys.modules, {"torch": torch, "whisper": whisper}):
            result = OpenAIBackend(model).decode(self.chunks(), language="ko")
        self.assertEqual([r.text for r in result], ["a", "b"])
        self.assertTrue(whisper.DecodingOptions.call_args.kwargs["without_timestamps"])
        model.transcribe.assert_not_called()
        self.assertEqual(whisper.log_mel_spectrogram.call_count, 2)

    def test_whisperx_vad_reuses_public_detector_and_merge_only(self):
        detector = Mock()
        detector.preprocess_audio.return_value = "wave"
        detector.return_value = "scores"
        detector.merge_chunks.return_value = [{"start": 0, "end": 1}]
        silero = Mock(return_value=detector)
        with patch.dict(sys.modules, {"whisperx.vads": SimpleNamespace(Silero=silero, Pyannote=Mock())}):
            vad = WhisperXVAD("silero", chunk_size=15)
            result = vad(np.zeros(RATE))
        self.assertEqual(result, [{"start": 0, "end": 1}])
        self.assertEqual(silero.call_args.kwargs["chunk_size"], 15)
        detector.merge_chunks.assert_called_once_with("scores", 15, onset=.5, offset=.363)

    def test_loader_rejects_reserved_options_without_importing_runtime(self):
        with self.assertRaisesRegex(ValueError, "Reserved"):
            load_backend("huggingface", "base", model_options={"device_map": "auto"})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            load_backend("typo", "base")
        with self.assertRaises(ValueError):
            DecodeConfig(0)


if __name__ == "__main__":
    unittest.main()
