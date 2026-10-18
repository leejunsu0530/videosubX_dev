"""Lazy-loaded adapters for benchmark.py; optional engines stay independent."""
from pathlib import Path


ENGINES = {
    "openai": ("cpu", "cuda"),
    "hf-pipeline": ("cpu", "cuda", "xpu"),
    "hf-direct": ("cpu", "cuda", "xpu"),
    "whispercpp": ("cpu",),
    "faster-whisper": ("cpu", "cuda"),
    "openvino-genai": ("cpu", "gpu", "npu"),
    "optimum-intel": ("cpu", "gpu"),
}


class Unsupported(RuntimeError):
    """Requested engine/device is unavailable; never silently fall back."""


def device_family(device):
    return device.lower().split(":")[0].split(".")[0]


def validate_device(engine, device):
    family = device_family(device)
    if family not in ENGINES[engine]:
        raise Unsupported(f"{engine} does not support {device}; choices: {ENGINES[engine]}")
    if family == "cpu" and device.lower() != "cpu":
        raise ValueError("CPU has no device index")
    if family in {"cuda", "xpu"}:
        # Explicit integer indices only: no auto placement or fallback.
        if device.lower() != family:
            prefix, index = device.lower().split(":")
            if prefix != family or not index.isdigit():
                raise ValueError(f"Invalid device: {device}")
    elif family in {"gpu", "npu"} and device.lower() != family:
        prefix, index = device.lower().split(".")
        if prefix != family or not index.isdigit():
            raise ValueError(f"Use {family}.0, {family}.1, ...")
    return family


class Backend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.engine = cfg["engine"]
        self.device = cfg["device"].lower()
        self.family = validate_device(self.engine, self.device)
        self.model_id = cfg["model"]
        self.load_options = dict(cfg.get("load_options", {}))
        self.decode_options = dict(cfg.get("decode_options", {}))
        self.language = cfg.get("language")
        self.beam = cfg.get("beam_size", 1)
        self.torch = None
        self.details = {}
        self._load()

    def _load(self):
        if self.engine in {"openai", "hf-pipeline", "hf-direct"}:
            import torch
            self.torch = torch
            if self.family != "cpu":
                api = getattr(torch, self.family, None)
                if api is None or not api.is_available():
                    raise Unsupported(f"PyTorch {self.device} is unavailable")
                index = int(self.device.split(":")[1]) if ":" in self.device else 0
                if index >= api.device_count():
                    raise Unsupported(f"No device {self.device}")
            torch.set_num_threads(self.cfg["threads"])
            dtype_name = self.cfg.get("dtype", "float32")
            if dtype_name not in {"float32", "float16", "bfloat16"}:
                raise ValueError("dtype must be float32, float16 or bfloat16")
            self.dtype = getattr(torch, dtype_name)
            self.details.update(dtype=dtype_name, device=self.device)
            if self.family in {"cuda", "xpu"}:
                self.details["device_name"] = getattr(torch, self.family).get_device_name(self.device)
            if self.engine == "openai":
                import whisper
                if dtype_name == "bfloat16" or (self.family == "cpu" and dtype_name != "float32"):
                    raise Unsupported("openai: use float32 on CPU, float32/float16 on CUDA")
                self.model = whisper.load_model(self.model_id, device=self.device, **self.load_options)
                return
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            if "/" not in self.model_id and not Path(self.model_id).exists():
                self.model_id = "openai/whisper-" + self.model_id
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            options = {"torch_dtype": self.dtype, "low_cpu_mem_usage": True, **self.load_options}
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id, **options)
            self.model.to(self.device).eval()
            if self.engine == "hf-pipeline":
                self.pipe = pipeline(
                    "automatic-speech-recognition", model=self.model,
                    tokenizer=self.processor.tokenizer,
                    feature_extractor=self.processor.feature_extractor, device=self.device,
                )
            return
        if self.engine == "faster-whisper":
            from faster_whisper import WhisperModel
            compute_type = self.cfg.get("compute_type", "float32")
            options = {"cpu_threads": self.cfg["threads"], **self.load_options}
            index = int(self.device.split(":")[1]) if ":" in self.device else 0
            self.model = WhisperModel(self.model_id, device=self.family, device_index=index,
                                      compute_type=compute_type, **options)
            self.details.update(compute_type=compute_type, device=self.device)
            return
        if self.engine == "whispercpp":
            from pywhispercpp.model import Model
            self.model = Model(
                self.model_id, context_params={"use_gpu": False},
                params_sampling_strategy=0 if self.beam == 1 else 1,
                n_threads=self.cfg["threads"], print_progress=False, print_realtime=False,
                **self.load_options,
            )
            self.details.update(device="cpu", precision="encoded in ggml model")
            return
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        device = self.device.upper()
        if not any(d == device or d.startswith(device + ".") for d in devices):
            raise Unsupported(f"OpenVINO {device} unavailable; detected: {devices}")
        self.details.update(device=device, available_devices=devices,
                            precision="encoded in exported IR and ov_config")
        self.details["device_names"] = {d: core.get_property(d, "FULL_DEVICE_NAME") for d in devices}
        # Local pre-exported models keep export/quantization out of timing.
        if not Path(self.model_id).is_dir():
            raise ValueError("OpenVINO engines need a local exported IR directory; see WHISPER_BENCHMARK.md")
        if self.engine == "openvino-genai":
            import openvino_genai as genai
            # New releases expose ASRPipeline; pinned 2026.x versions use WhisperPipeline.
            pipeline_cls = getattr(genai, "ASRPipeline", None)
            if pipeline_cls is None:
                pipeline_cls = genai.WhisperPipeline
            options = dict(self.load_options)
            if self.family == "cpu":
                options.setdefault("INFERENCE_NUM_THREADS", self.cfg["threads"])
            self.model = pipeline_cls(self.model_id, device, **options)
            self.details["pipeline_class"] = pipeline_cls.__name__
        else:
            from optimum.intel import OVModelForSpeechSeq2Seq
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            options = dict(self.load_options)
            if self.family == "cpu":
                options["ov_config"] = {"INFERENCE_NUM_THREADS": self.cfg["threads"],
                                        **options.get("ov_config", {})}
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, device=device, compile=True, **options,
            )

    def sync(self):
        if self.torch is not None and self.family in {"cuda", "xpu"}:
            getattr(self.torch, self.family).synchronize(self.device)

    def reset_device_memory(self):
        if self.torch is not None and self.family == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def device_memory(self):
        if self.torch is not None and self.family == "cuda":
            return {"torch_cuda_peak_allocated_mib": self.torch.cuda.max_memory_allocated(self.device) / 2**20,
                    "torch_cuda_peak_reserved_mib": self.torch.cuda.max_memory_reserved(self.device) / 2**20}
        return {"torch_cuda_peak_allocated_mib": None, "torch_cuda_peak_reserved_mib": None}

    def transcribe(self, audio):
        if self.engine == "openai":
            options = {"language": self.language, "task": "transcribe", "temperature": 0,
                       "fp16": self.cfg.get("dtype", "float32") == "float16", "verbose": None}
            if self.beam > 1:
                options["beam_size"] = self.beam
            return self.model.transcribe(audio, **{**options, **self.decode_options})["text"]
        if self.engine == "faster-whisper":
            segments, _ = self.model.transcribe(audio, **{
                "language": self.language, "task": "transcribe", "beam_size": self.beam,
                "temperature": 0, "vad_filter": False, **self.decode_options,
            })
            # Generator iteration MUST be inside the timed region.
            return "".join(segment.text for segment in segments)
        if self.engine == "whispercpp":
            options = {"language": self.language or "auto", "translate": False,
                       "temperature": 0.0, "temperature_inc": 0.0}
            if self.beam > 1:
                options["beam_search"] = {"beam_size": self.beam, "patience": -1.0}
            # pywhispercpp strips each segment, so concatenate with explicit word boundaries.
            return " ".join(s.text.strip() for s in self.model.transcribe(audio, **{**options, **self.decode_options}))
        options = {"task": "transcribe", "num_beams": self.beam}
        if self.language:
            options["language"] = self.language
        if self.engine == "openvino-genai":
            if self.language:
                options["language"] = f"<|{self.language}|>"
            result = self.model.generate(audio.tolist(), **{**options, **self.decode_options})
            return "".join(result.texts)
        # Sequential long-form generation avoids silently truncating audio at 30s.
        options["return_timestamps"] = True
        options.update(self.decode_options)
        if self.engine == "hf-pipeline":
            with self.torch.inference_mode():
                return self.pipe({"raw": audio, "sampling_rate": 16000},
                                 return_timestamps=True, generate_kwargs=options)["text"]
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt",
                                truncation=False, padding="longest", return_attention_mask=True)
        if self.torch is not None:
            inputs = inputs.to(self.device)
            inputs["input_features"] = inputs["input_features"].to(self.dtype)
            with self.torch.inference_mode():
                ids = self.model.generate(**inputs, **options)
        else:
            ids = self.model.generate(**inputs, **options)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]
