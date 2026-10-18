"""VAD chunks on the original timeline, independent of the Whisper runtime."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
from typing import Callable, Iterable, Mapping, Protocol, Sequence
import warnings

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 30 * SAMPLE_RATE


@dataclass(frozen=True)
class Chunk:
    """Right-zero-padded mono float32 audio; valid_samples excludes padding."""

    audio: np.ndarray
    valid_samples: int
    start: float
    end: float


@dataclass(frozen=True)
class Decoded:
    text: str
    avg_logprob: float | None = None


class Backend(Protocol):
    name: str
    supports_batch: bool

    def decode(self, chunks: Sequence[Chunk], *, language: str) -> Sequence[Decoded]: ...


class SpeechDetector(Protocol):
    def __call__(self, audio: np.ndarray) -> Iterable[Mapping[str, float]]: ...


def load_audio(path: str | Path) -> np.ndarray:
    """Decode once to mono 16 kHz. Array inputs to transcribe must already match."""
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
               "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
               "-ar", str(SAMPLE_RATE), "-"]
    try:
        result = subprocess.run(command, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.decode(errors="replace")) from exc
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0


def merge_regions(regions, duration: float, chunk_size: float = 30) -> list[dict]:
    """Greedy WhisperX-style merge by elapsed span, retaining internal silence.

    Custom VADs must split long speech before returning regions. We reject long
    regions rather than truncate audio or invent a cut through a spoken word.
    """
    if not math.isfinite(chunk_size) or not 0 < chunk_size <= 30:
        raise ValueError("chunk_size must be in (0, 30]")
    normalized = []
    previous_end = 0.0
    for region in regions:
        start, end = float(region["start"]), float(region["end"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"Invalid speech region: {region}")
        # Some detectors put the last boundary a fraction of a frame past EOF.
        end = min(end, duration)
        if start >= end:
            raise ValueError("Speech region starts outside the audio")
        if start < previous_end - 1 / SAMPLE_RATE:
            raise ValueError("Speech regions must be ordered and non-overlapping")
        if end - start > chunk_size + 1 / SAMPLE_RATE:
            raise ValueError("VAD region exceeds chunk_size; split it in the detector")
        normalized.append({"start": start, "end": end})
        previous_end = end
    merged = []
    for region in normalized:
        if merged and region["end"] - merged[-1]["start"] <= chunk_size:
            merged[-1]["end"] = region["end"]
        else:
            merged.append(dict(region))
    return merged


class WhisperXVAD:
    """Reuse WhisperX VAD and its min-cut/merge, without loading its ASR model."""

    def __init__(self, method="pyannote", *, device="cpu", chunk_size=30,
                 onset=0.5, offset=0.363, **model_options):
        if not 0 < chunk_size <= 30:
            raise ValueError("chunk_size must be in (0, 30]")
        if method not in {"pyannote", "silero"}:
            raise ValueError("VAD method must be pyannote or silero")
        # Lazy import: custom/precomputed VAD does not need WhisperX or torch.
        from whisperx.vads import Pyannote, Silero
        options = {**model_options, "chunk_size": chunk_size,
                   "vad_onset": onset, "vad_offset": offset}
        if method == "pyannote":
            import torch
            self.model = Pyannote(torch.device(device), **options)
        else:
            self.model = Silero(**options)
        self.chunk_size, self.onset, self.offset = chunk_size, onset, offset

    def __call__(self, audio):
        scores = self.model({"waveform": self.model.preprocess_audio(audio),
                             "sample_rate": SAMPLE_RATE})
        return self.model.merge_chunks(scores, self.chunk_size,
                                       onset=self.onset, offset=self.offset)


class TranscriptionPipeline:
    def __init__(self, backend: Backend, vad: SpeechDetector | None = None, *,
                 chunk_size=30):
        if not math.isfinite(chunk_size) or not 0 < chunk_size <= 30:
            raise ValueError("chunk_size must be in (0, 30]")
        self.backend, self.vad, self.chunk_size = backend, vad, chunk_size

    def transcribe(self, audio: str | Path | np.ndarray, *, language: str,
                   batch_size: int = 8, speech_regions=None,
                   progress_callback: Callable[[float], None] | None = None) -> dict:
        """Return WhisperX-compatible segments. Language is explicit, once per file.

        speech_regions overrides VAD and can be cached for fair engine comparisons.
        Neither silence compression nor cross-chunk text context is performed.
        """
        if not isinstance(language, str) or not language.strip() or language == "auto":
            raise ValueError("Provide an explicit language code, e.g. 'ko' or 'en'")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(audio, (str, Path)):
            audio = load_audio(audio)
        audio = np.asarray(audio)
        if audio.ndim != 1 or not np.issubdtype(audio.dtype, np.floating):
            raise ValueError("Audio must be mono floating-point samples at 16000 Hz")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains non-finite samples")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if speech_regions is None:
            if self.vad is None:
                raise ValueError("Provide a VAD detector or speech_regions (use [] for silence)")
            speech_regions = self.vad(audio) if audio.size else []
        regions = merge_regions(speech_regions, len(audio) / SAMPLE_RATE, self.chunk_size)
        step = batch_size
        if not self.backend.supports_batch:
            step = 1
            if batch_size > 1 and regions:
                warnings.warn(f"{self.backend.name} has no cross-chunk batch API; decoding sequentially",
                              UserWarning, stacklevel=2)
        segments = []
        for offset in range(0, len(regions), step):
            chunks = []
            for region in regions[offset:offset + step]:
                f1, f2 = (int(region[key] * SAMPLE_RATE) for key in ("start", "end"))
                waveform = audio[f1:f2]
                if not 0 < len(waveform) <= WINDOW_SAMPLES:
                    raise ValueError("Speech chunk must contain 1..480000 samples")
                padded = np.pad(waveform, (0, WINDOW_SAMPLES - len(waveform)))
                chunks.append(Chunk(padded, len(waveform), region["start"], region["end"]))
            outputs = list(self.backend.decode(chunks, language=language))
            if len(outputs) != len(chunks):
                raise RuntimeError("Backend must return exactly one result per chunk, in input order")
            for chunk, output in zip(chunks, outputs):
                if not isinstance(output, Decoded) or not isinstance(output.text, str):
                    raise TypeError("Backend results must be Decoded objects containing text")
                segment = {"start": chunk.start, "end": chunk.end, "text": output.text}
                if output.avg_logprob is not None:
                    segment["avg_logprob"] = output.avg_logprob
                segments.append(segment)
            if progress_callback is not None:
                progress_callback(len(segments) / len(regions) * 100)
        if not regions and progress_callback is not None:
            progress_callback(100.0)
        return {"segments": segments, "language": language}


def align_result(result: dict, audio, *, device="cpu", model_name=None,
                 model_dir=None, return_char_alignments=False) -> dict:
    """Align against original, unpadded audio; preserves the detected language."""
    if not result["segments"]:
        return {**result, "word_segments": []}
    import whisperx
    if isinstance(audio, Path):
        audio = str(audio)
    model, metadata = whisperx.load_align_model(result["language"], device,
                                               model_name=model_name, model_dir=model_dir)
    aligned = whisperx.align(result["segments"], model, metadata, audio, device,
                            return_char_alignments=return_char_alignments)
    return {**aligned, "language": result["language"]}
