"""Swappable short-form Whisper transcription with shared VAD and alignment."""
from .backends import (DecodeConfig, FasterWhisperBackend, OpenAIBackend,
                       TransformersBackend, WhisperCppBackend, load_backend,
                       register_backend)
from .pipeline import (Backend, Chunk, Decoded, SpeechDetector, TranscriptionPipeline,
                       WhisperXVAD, align_result, load_audio, merge_regions)

__all__ = ["Backend", "Chunk", "Decoded", "DecodeConfig", "SpeechDetector",
           "TranscriptionPipeline", "WhisperXVAD", "align_result", "load_audio",
           "merge_regions", "load_backend", "register_backend", "OpenAIBackend",
           "FasterWhisperBackend", "TransformersBackend", "WhisperCppBackend"]
