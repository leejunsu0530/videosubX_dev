"""Short-form adapters. Optional runtimes are imported only by their factory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pipeline import Decoded, SAMPLE_RATE, WINDOW_SAMPLES


@dataclass(frozen=True)
class DecodeConfig:
    beam_size: int = 5
    initial_prompt: str | None = None

    def __post_init__(self):
        if type(self.beam_size) is not int or self.beam_size < 1:
            raise ValueError("beam_size must be a positive integer")
        if self.initial_prompt is not None and not isinstance(self.initial_prompt, str):
            raise ValueError("initial_prompt must be text or None")


class OpenAIBackend:
    name, supports_batch = "openai", True

    def __init__(self, model, *, config=DecodeConfig(), fp16=False):
        self.model, self.config, self.fp16 = model, config, fp16

    def decode(self, chunks, *, language):
        import torch
        import whisper
        if not self.model.is_multilingual and language != "en":
            raise ValueError("English-only model requires language='en'")
        # Use decode(), not the sequential long-form transcribe() orchestration.
        mel = torch.stack([whisper.log_mel_spectrogram(c.audio, self.model.dims.n_mels)
                           for c in chunks]).to(self.model.device)
        options = whisper.DecodingOptions(language=language, task="transcribe",
                                         without_timestamps=True, temperature=0,
                                         beam_size=self.config.beam_size,
                                         prompt=self.config.initial_prompt, fp16=self.fp16)
        with torch.inference_mode():
            results = whisper.decode(self.model, mel, options)
        return [Decoded(r.text, r.avg_logprob) for r in results]


class TransformersBackend:
    """Shared generate() path for HF PyTorch, Optimum OpenVINO and ONNX Runtime."""

    supports_batch = True

    def __init__(self, model, processor, *, name="huggingface", config=DecodeConfig(),
                 device=None, dtype=None):
        self.model, self.processor, self.name = model, processor, name
        self.config, self.device, self.dtype = config, device, dtype

    def decode(self, chunks, *, language):
        import torch
        # Let the processor pad the original samples so attention_mask still
        # records real lengths. The waveform padding target is exactly 30s.
        inputs = self.processor([c.audio[:c.valid_samples] for c in chunks],
                                sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                padding="max_length", max_length=WINDOW_SAMPLES,
                                truncation=False, return_attention_mask=True)
        if self.device is not None:
            inputs = inputs.to(self.device)
            inputs["input_features"] = inputs["input_features"].to(self.dtype)
        options = {"language": language, "task": "transcribe", "return_timestamps": False,
                   "return_token_timestamps": False, "do_sample": False,
                   "num_beams": self.config.beam_size,
                   "condition_on_prev_tokens": False}
        if getattr(self.model.generation_config, "is_multilingual", True) is False:
            if language != "en":
                raise ValueError("English-only model requires language='en'")
            options.pop("language")
            options.pop("task")
        if self.config.initial_prompt:
            options["prompt_ids"] = self.processor.get_prompt_ids(
                self.config.initial_prompt, return_tensors="pt")
            if self.device is not None:
                options["prompt_ids"] = options["prompt_ids"].to(self.device)
            options["prompt_condition_type"] = "first-segment"
        with torch.inference_mode():
            ids = self.model.generate(**inputs, **options)
        return [Decoded(text) for text in self.processor.batch_decode(ids, skip_special_tokens=True)]


class FasterWhisperBackend:
    name, supports_batch = "faster-whisper", True

    def __init__(self, model, *, config=DecodeConfig()):
        self.model, self.config = model, config

    def decode(self, chunks, *, language):
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import get_suppressed_tokens
        if not self.model.model.is_multilingual and language != "en":
            raise ValueError("English-only model requires language='en'")
        tokenizer = Tokenizer(self.model.hf_tokenizer, self.model.model.is_multilingual,
                              task="transcribe", language=language)
        prompt_tokens = (tokenizer.encode(" " + self.config.initial_prompt.strip())
                         if self.config.initial_prompt else [])
        prompt = self.model.get_prompt(tokenizer, prompt_tokens, without_timestamps=True)
        # Avoid feature-extractor tail padding: waveforms already contain 30s.
        # This keeps exactly 3000 Mel frames rather than an extra frame.
        features = np.stack([self.model.feature_extractor(c.audio, padding=0) for c in chunks])
        encoded = self.model.encode(features)
        outputs = self.model.model.generate(
            encoded, [list(prompt) for _ in chunks], beam_size=self.config.beam_size,
            patience=1, length_penalty=1, max_length=self.model.max_length,
            suppress_blank=True, suppress_tokens=get_suppressed_tokens(tokenizer, [-1]),
            return_scores=True,
        )
        results = []
        for output in outputs:
            ids = output.sequences_ids[0]
            text = tokenizer.decode([t for t in ids if t < tokenizer.eot])
            avg = output.scores[0] * len(ids) / (len(ids) + 1)
            results.append(Decoded(text, avg))
        return results


class WhisperCppBackend:
    name, supports_batch = "whispercpp", False

    def __init__(self, model, *, config=DecodeConfig()):
        self.model, self.config = model, config

    def decode(self, chunks, *, language):
        options = {"language": language, "translate": False, "no_context": True,
                   "no_timestamps": True, "token_timestamps": False,
                   "single_segment": True, "temperature": 0.0, "temperature_inc": 0.0,
                   "vad": False, "greedy": {"best_of": 1},
                   "initial_prompt": self.config.initial_prompt or "",
                   "print_progress": False, "print_realtime": False}
        if self.config.beam_size > 1:
            options["beam_search"] = {"beam_size": self.config.beam_size, "patience": -1.0}
        return [Decoded(" ".join(s.text.strip() for s in self.model.transcribe(c.audio, **options)))
                for c in chunks]


_BUILTINS = {"openai", "huggingface", "faster-whisper", "whispercpp",
             "optimum-intel", "optimum-onnx"}
_FACTORIES = {}


def register_backend(name, factory):
    """Register a factory accepting model_id and load_backend's keyword arguments."""
    if not isinstance(name, str) or not name or name in _FACTORIES or name in _BUILTINS:
        raise ValueError(f"Backend name is empty or already registered: {name}")
    _FACTORIES[name] = factory


def load_backend(engine: str, model_id: str, *, device="cpu", beam_size=5,
                 initial_prompt=None, dtype="float32", compute_type="default",
                 model_options=None):
    """Load one runtime. model_options are constructor options, not decode overrides.

    Use exported local models for Optimum. Unsupported configurations raise;
    no automatic switch of engine/device or sequential retry is performed.
    """
    config = DecodeConfig(beam_size, initial_prompt)
    options = dict(model_options or {})
    reserved = {"device", "device_index", "device_map", "dtype", "torch_dtype",
                "compute_type", "provider", "provider_options", "context_params",
                "params_sampling_strategy", "use_openvino", "export", "compile"}
    if reserved & options.keys():
        raise ValueError(f"Reserved model options: {sorted(reserved & options.keys())}")
    device = device.lower()
    if engine in _FACTORIES:
        return _FACTORIES[engine](model_id, device=device, config=config, dtype=dtype,
                                  compute_type=compute_type, model_options=options)
    if engine == "whispercpp":
        if device != "cpu":
            raise ValueError("Built-in whispercpp loader supports CPU; inject a GPU model explicitly")
        from pywhispercpp.model import Model
        # Values passed through **params persist between calls in pywhispercpp.
        for key in ("no_context", "no_timestamps", "token_timestamps", "single_segment",
                    "temperature", "temperature_inc", "initial_prompt", "language",
                    "vad", "audio_ctx", "offset_ms", "duration_ms"):
            if key in options:
                raise ValueError(f"Decode option cannot be set in model_options: {key}")
        options.setdefault("print_progress", False)
        options.setdefault("print_realtime", False)
        model = Model(model_id, context_params={"use_gpu": False, "dtw_token_timestamps": False},
                      params_sampling_strategy=0 if beam_size == 1 else 1,
                      **options)
        return WhisperCppBackend(model, config=config)
    if engine == "faster-whisper":
        if device != "cpu" and device != "cuda" and not (
                device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("faster-whisper device must be cpu or cuda[:index]")
        from faster_whisper import WhisperModel
        family, _, index = device.partition(":")
        model = WhisperModel(model_id, device=family, device_index=int(index or 0),
                             compute_type=compute_type, **options)
        return FasterWhisperBackend(model, config=config)
    if engine in {"openai", "huggingface"}:
        import torch
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16 or bfloat16")
        target = torch.device(device)
        if target.type not in {"cpu", "cuda", "xpu", "mps"}:
            raise ValueError("Unsupported PyTorch device")
        if target.type != "cpu":
            api = getattr(torch, target.type)
            if not api.is_available():
                raise ValueError(f"PyTorch device unavailable: {device}")
        if engine == "openai":
            if dtype == "bfloat16" or (target.type == "cpu" and dtype != "float32"):
                raise ValueError("openai requires float32 on CPU and float32/float16 otherwise")
            import whisper
            return OpenAIBackend(whisper.load_model(model_id, device=device, **options),
                                 config=config, fp16=dtype == "float16")
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        model_id = _hf_id(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), **options).to(device).eval()
        return TransformersBackend(model, processor, config=config,
                                   device=device, dtype=getattr(torch, dtype))
    if engine in {"optimum-intel", "optimum-onnx"}:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        if engine == "optimum-intel":
            if device not in {"cpu", "gpu"} and not (device.startswith("gpu.") and device[4:].isdigit()):
                raise ValueError("optimum-intel device must be cpu or gpu[.index]")
            from optimum.intel import OVModelForSpeechSeq2Seq
            model = OVModelForSpeechSeq2Seq.from_pretrained(
                model_id, device=device.upper(), compile=True, **options)
        else:
            if device != "cpu" and device != "cuda" and not (
                    device.startswith("cuda:") and device[5:].isdigit()):
                raise ValueError("optimum-onnx device must be cpu or cuda[:index]")
            from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
            provider = "CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider"
            import onnxruntime as ort
            if provider not in ort.get_available_providers():
                raise ValueError(f"ONNX provider unavailable: {provider}")
            provider_options = {} if device == "cpu" else {"device_id": int(device.partition(":")[2] or 0)}
            model = ORTModelForSpeechSeq2Seq.from_pretrained(
                model_id, provider=provider, provider_options=provider_options, **options)
            if provider not in model.providers:
                raise RuntimeError(f"ONNX Runtime fell back from requested provider {provider}")
        return TransformersBackend(model, processor, name=engine, config=config)
    raise ValueError(f"Unknown engine: {engine}")


def _hf_id(model_id):
    if "/" not in model_id and not Path(model_id).exists():
        return "openai/whisper-" + model_id
    return model_id
