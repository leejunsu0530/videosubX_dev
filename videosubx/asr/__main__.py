"""python -m videosubx.asr input.wav --engine whispercpp --model base --language ko"""
import argparse
import gc
import json
from pathlib import Path

from . import TranscriptionPipeline, WhisperXVAD, align_result, load_audio, load_backend


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shared WhisperX VAD with a swappable Whisper runtime")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--engine", required=True, choices=["openai", "huggingface", "faster-whisper",
                                                          "whispercpp", "optimum-intel", "optimum-onnx"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", required=True, help="Explicit language code, e.g. ko")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--compute-type", default="default", help="faster-whisper compute type")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--initial-prompt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=float, default=30)
    parser.add_argument("--vad", default="pyannote", choices=["pyannote", "silero"])
    parser.add_argument("--vad-device", default="cpu")
    parser.add_argument("--speech-regions", type=Path, help="Cached JSON list of start/end seconds; skips VAD")
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--align-device", default="cpu")
    parser.add_argument("--align-model")
    parser.add_argument("--output", type=Path, default=Path("transcript.json"))
    args = parser.parse_args(argv)
    audio = load_audio(args.audio)
    regions = json.loads(args.speech_regions.read_text(encoding="utf-8-sig")) if args.speech_regions else None
    vad = None if regions is not None else WhisperXVAD(args.vad, device=args.vad_device,
                                                     chunk_size=args.chunk_size)
    backend = load_backend(args.engine, args.model, device=args.device, dtype=args.dtype,
                           compute_type=args.compute_type, beam_size=args.beam_size,
                           initial_prompt=args.initial_prompt)
    pipeline = TranscriptionPipeline(backend, vad, chunk_size=args.chunk_size)
    result = pipeline.transcribe(audio, language=args.language, batch_size=args.batch_size,
                                 speech_regions=regions)
    # Drop ASR/VAD references before loading the separate alignment model.
    del pipeline, backend, vad
    gc.collect()
    if args.align:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = align_result(result, audio, device=args.align_device, model_name=args.align_model)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
