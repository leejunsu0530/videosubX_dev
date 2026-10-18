"""Whisper benchmark CLI. See WHISPER_BENCHMARK.md; no engines imported by controller."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import unicodedata

from whisper_backends import Backend, ENGINES, Unsupported, validate_device

DEFAULTS = {"model": "base", "device": "cpu", "language": None, "threads": 4,
            "beam_size": 1, "warmup": 1, "repeats": 3, "dtype": "float32",
            "compute_type": "float32", "normalization": "basic", "sample_interval": 0.02,
            "timeout": 3600, "ffmpeg": "ffmpeg", "load_options": {}, "decode_options": {}}
PACKAGES = ["numpy", "psutil", "rapidfuzz", "torch", "transformers", "openai-whisper", "faster-whisper",
            "ctranslate2", "pywhispercpp", "openvino", "openvino-genai", "optimum-intel"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def normalize(text, mode):
    if mode == "none":
        return text
    text = unicodedata.normalize("NFKC", text).casefold()
    # Strip punctuation without inserting spaces inside contractions/decimal numbers.
    text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return " ".join(text.split())


def edit_distance(reference, hypothesis):
    """Levenshtein with O(min(N,M)) space; counts substitutions/deletions/insertions equally."""
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        pass  # Offline self-tests can run without common worker dependencies.
    else:
        return Levenshtein.distance(reference, hypothesis)
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    row = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        next_row = [i]
        for j, hyp in enumerate(hypothesis, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (ref != hyp)))
        row = next_row
    return row[-1]


def error_rate(errors, units):
    return errors / units if units else (0.0 if errors == 0 else None)


def score(reference, hypothesis, mode):
    if reference is None:
        return {"wer": None, "cer": None}
    ref, hyp = normalize(reference, mode), normalize(hypothesis, mode)
    words_r, words_h = ref.split(), hyp.split()
    chars_r, chars_h = "".join(ref.split()), "".join(hyp.split())
    we, ce = edit_distance(words_r, words_h), edit_distance(chars_r, chars_h)
    return {"wer": error_rate(we, len(words_r)), "cer": error_rate(ce, len(chars_r)),
            "word_errors": we, "reference_words": len(words_r),
            "character_errors": ce, "reference_characters": len(chars_r)}


def resolve_sample(sample, base):
    if not isinstance(sample, dict) or "audio" not in sample:
        raise ValueError("Each sample needs an audio path")
    unknown = set(sample) - {"audio", "reference", "reference_file"}
    if unknown:
        raise ValueError(f"Unknown sample keys: {sorted(unknown)}")
    result = dict(sample)
    audio = (base / sample["audio"]).resolve()
    if not audio.is_file():
        raise ValueError(f"Audio does not exist: {audio}")
    result["audio"] = str(audio)
    if "reference_file" in sample:
        if "reference" in sample:
            raise ValueError("Use reference OR reference_file")
        result["reference"] = (base / sample["reference_file"]).read_text(encoding="utf-8-sig")
        del result["reference_file"]
    if result.get("reference") is not None and not isinstance(result["reference"], str):
        raise ValueError("reference must be text or null")
    return result


def validate_run(cfg):
    unknown = set(cfg) - set(DEFAULTS) - {"engine", "python", "samples", "id"}
    if unknown:
        raise ValueError(f"Unknown run keys: {sorted(unknown)}")
    if cfg.get("engine") not in ENGINES:
        raise ValueError(f"Unknown engine: {cfg.get('engine')}")
    for key in ("threads", "beam_size", "repeats", "warmup"):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "warmup" else 1):
            raise ValueError(f"Invalid {key}: {cfg[key]}")
    for key in ("sample_interval", "timeout"):
        if not isinstance(cfg[key], (float, int)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if cfg["normalization"] not in {"basic", "none"}:
        raise ValueError("normalization must be basic or none")
    if not isinstance(cfg["model"], str) or not cfg["model"]:
        raise ValueError("model must be a nonempty string")
    if cfg["language"] is not None and not isinstance(cfg["language"], str):
        raise ValueError("language must be an ISO code or null")
    if not isinstance(cfg["device"], str):
        raise ValueError("device must be a string")
    for key in ("load_options", "decode_options"):
        if not isinstance(cfg[key], dict):
            raise ValueError(f"{key} must be an object")
    reserved = {"device", "device_index", "device_map", "torch_dtype", "dtype", "compute_type",
                "context_params", "use_openvino", "export", "cpu_threads", "n_threads",
                "params_sampling_strategy", "compile"}
    if reserved & cfg["load_options"].keys():
        raise ValueError(f"Use top-level settings instead of load_options: {reserved & cfg['load_options'].keys()}")
    if not cfg.get("samples"):
        raise ValueError("At least one audio sample is required")


class MemorySampler:
    """Sample worker + live descendants RSS. Shared pages may be counted twice."""
    def __init__(self, interval):
        import psutil
        self.psutil = psutil
        self.process = psutil.Process()
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak = self.current()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def current(self):
        total = 0
        for process in [self.process, *self.process.children(recursive=True)]:
            try:
                total += process.memory_info().rss
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                pass
        return total / 2**20

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self.peak = max(self.peak, self.current())

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.current())


def load_audio(path, ffmpeg):
    import numpy as np
    command = [ffmpeg, "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0", "-vn",
               "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    return audio


def metadata():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(), "packages": versions}


def worker(cfg):
    result = {"config": cfg, "environment": metadata(), "rows": [], "status": "error"}
    try:
        validate_device(cfg["engine"], cfg["device"])
        baseline = MemorySampler(cfg["sample_interval"]).current()
        started = time.perf_counter()
        with MemorySampler(cfg["sample_interval"]) as memory:
            backend = Backend(cfg)
            backend.sync()
        result.update(load_seconds=time.perf_counter() - started,
                      baseline_rss_mib=baseline, load_peak_rss_mib=memory.peak,
                      loaded_rss_mib=memory.current(), backend=backend.details,
                      resolved_model=backend.model_id)
        for index, sample in enumerate(cfg["samples"]):
            started = time.perf_counter()
            with MemorySampler(cfg["sample_interval"]) as memory:
                audio = load_audio(sample["audio"], cfg["ffmpeg"])
            decode_seconds = time.perf_counter() - started
            duration = len(audio) / 16000
            result.setdefault("audio_preparation", []).append({
                "sample": index, "seconds": decode_seconds, "peak_rss_mib": memory.peak})
            with MemorySampler(cfg["sample_interval"]) as memory:
                for _ in range(cfg["warmup"]):
                    backend.transcribe(audio)
                    backend.sync()
            result.setdefault("warmup_peak_rss_mib", []).append(memory.peak)
            for repeat in range(cfg["repeats"]):
                backend.sync()
                backend.reset_device_memory()
                before = memory.current()
                with MemorySampler(cfg["sample_interval"]) as memory:
                    started = time.perf_counter()
                    text = backend.transcribe(audio)
                    backend.sync()
                    elapsed = time.perf_counter() - started
                result["rows"].append({
                    "sample": index, "audio": sample["audio"], "repeat": repeat,
                    "audio_seconds": duration, "seconds": elapsed, "rtf": elapsed / duration,
                    "audio_seconds_per_second": duration / elapsed, "text": text,
                    "reference": sample.get("reference"), "peak_rss_mib": memory.peak,
                    "rss_before_mib": before, "rss_peak_increase_mib": max(0, memory.peak - before),
                    **backend.device_memory(), **score(sample.get("reference"), text, cfg["normalization"]),
                })
            del audio
        result["status"] = "ok"
    except Unsupported as exc:
        result.update(status="skipped", reason=str(exc))
    except ModuleNotFoundError as exc:
        result.update(status="skipped", reason=f"Missing dependency: {exc}")
    except Exception as exc:
        result.update(status="error", reason=str(exc), traceback=traceback.format_exc())
    if result["rows"]:
        result["summary"] = summarize(result["rows"])
    return result


def summarize(rows):
    times = [r["seconds"] for r in rows]
    # Accuracy is corpus-weighted, one transcript per sample (first measured repeat).
    scored = [r for r in rows if r["repeat"] == 0 and "word_errors" in r]
    def corpus(prefix, unit):
        return error_rate(sum(r[prefix] for r in scored), sum(r[unit] for r in scored)) if scored else None
    return {"inferences": len(rows), "mean_seconds": statistics.mean(times),
            "median_seconds": statistics.median(times),
            "stdev_seconds": statistics.stdev(times) if len(times) > 1 else 0,
            "total_inference_seconds": sum(times),
            "rtf": sum(times) / sum(r["audio_seconds"] for r in rows),
            "inference_peak_rss_mib": max(r["peak_rss_mib"] for r in rows),
            "corpus_wer": corpus("word_errors", "reference_words"),
            "corpus_cer": corpus("character_errors", "reference_characters"),
            "scored_samples": len(scored)}


def build_runs(args):
    if args.config:
        path = Path(args.config).resolve()
        data = read_json(path)
        if set(data) - {"defaults", "samples", "runs"}:
            raise ValueError("Config keys: defaults, samples, runs")
        if not data.get("runs"):
            raise ValueError("Config requires a nonempty runs list")
        runs = []
        for override in data["runs"]:
            cfg = {**DEFAULTS, **data.get("defaults", {}), **override}
            cfg["samples"] = [resolve_sample(s, path.parent) for s in cfg.get("samples", data.get("samples", []))]
            # Config paths are relative to the JSON file, HF IDs stay literal.
            if (path.parent / cfg["model"]).exists():
                cfg["model"] = str((path.parent / cfg["model"]).resolve())
            if cfg.get("python"):
                candidate = path.parent / cfg["python"]
                if candidate.exists():
                    cfg["python"] = str(candidate.resolve())
            runs.append(cfg)
    else:
        samples = []
        if args.manifest:
            path = Path(args.manifest).resolve()
            samples.extend(resolve_sample(json.loads(line), path.parent)
                           for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
        samples.extend(resolve_sample({"audio": a}, Path.cwd()) for a in args.audio or [])
        if args.reference:
            if len(samples) != 1:
                raise ValueError("--reference needs exactly one sample; use --manifest for a corpus")
            samples[0]["reference"] = Path(args.reference).read_text(encoding="utf-8-sig")
        overrides = {key: getattr(args, key) for key in DEFAULTS if hasattr(args, key) and getattr(args, key) is not None}
        runs = [{**DEFAULTS, **overrides, "engine": engine, "device": device, "samples": samples}
                for engine in args.engine for device in args.device]
    for cfg in runs:
        validate_run(cfg)
    return runs


def export_results(output, results):
    write_json(output / "results.json", {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                                         "runs": results})
    summaries, rows = [], []
    for result in results:
        cfg = result["config"]
        identity = {"id": result["id"], **{k: cfg[k] for k in ("engine", "model", "device")}}
        summaries.append({**identity, "status": result["status"], "reason": result.get("reason"),
                          "load_seconds": result.get("load_seconds"),
                          "load_peak_rss_mib": result.get("load_peak_rss_mib"), **result.get("summary", {})})
        rows.extend({**identity, **r} for r in result.get("rows", []))
    for name, records in (("summary.csv", summaries), ("transcriptions.csv", rows)):
        fields = list(dict.fromkeys(k for record in records for k in record))
        with (output / name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)


def run_isolated(cfg, directory):
    request, response = directory / "request.json", directory / "result.json"
    write_json(request, cfg)
    env = {**os.environ, "OMP_NUM_THREADS": str(cfg["threads"]), "MKL_NUM_THREADS": str(cfg["threads"]),
           "PYTHONIOENCODING": "utf-8"}
    command = [cfg.get("python", sys.executable), str(Path(__file__).resolve()),
               "--worker", str(request), str(response)]
    started = time.perf_counter()
    try:
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                       start_new_session=os.name != "nt")
            try:
                process.wait(timeout=cfg["timeout"])
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                # Kill decoder subprocesses as well, including ffmpeg on Windows.
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        if response.exists():
            result = read_json(response)
        else:
            result = {"status": "error", "reason": f"Worker exited {process.returncode}; see worker.log", "rows": []}
    except subprocess.TimeoutExpired:
        result = {"status": "error", "reason": f"Timeout after {cfg['timeout']} seconds", "rows": []}
    except OSError as exc:
        result = {"status": "error", "reason": str(exc), "rows": []}
    return {**result, "config": cfg, "wall_seconds": time.perf_counter() - started, "log": str(directory / "worker.log")}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="JSON matrix; paths relative to config file")
    p.add_argument("--engine", nargs="+", choices=ENGINES, default=list(ENGINES))
    p.add_argument("--model", default=None)
    p.add_argument("--device", nargs="+", default=["cpu"], help="cpu, cuda:0, xpu:0, gpu.0, npu")
    p.add_argument("--audio", nargs="+")
    p.add_argument("--reference", help="UTF-8 plain transcript for one audio file")
    p.add_argument("--manifest", help="JSONL corpus with audio/reference/reference_file")
    p.add_argument("--language", help="ISO language code, e.g. ko, en; omitted = autodetect")
    for key in ("threads", "beam_size", "warmup", "repeats"):
        p.add_argument("--" + key.replace("_", "-"), type=int)
    for key in ("timeout", "sample_interval"):
        p.add_argument("--" + key.replace("_", "-"), type=float)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--compute-type", help="CTranslate2 precision, e.g. float32, int8, float16")
    p.add_argument("--normalization", choices=["basic", "none"])
    p.add_argument("--ffmpeg", help="ffmpeg executable path")
    p.add_argument("--output", help="New output directory (existing nonempty directory is refused)")
    p.add_argument("--list-engines", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Validate and print resolved matrix; no models loaded")
    p.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.worker:
        request, response = args.worker
        write_json(response, worker(read_json(request)))
        return 0
    if args.list_engines:
        for engine, devices in ENGINES.items():
            print(f"{engine:18} {', '.join(devices)}")
        return 0
    try:
        runs = build_runs(args)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        p.error(str(exc))
    if args.dry_run:
        print(json.dumps(runs, ensure_ascii=False, indent=2))
        return 0
    output = Path(args.output or f"benchmark-results/{datetime.now():%Y%m%d-%H%M%S-%f}").resolve()
    if output.exists() and any(output.iterdir()):
        p.error(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, cfg in enumerate(runs):
        job = f"{index:03d}-{cfg['engine']}"
        directory = output / job
        directory.mkdir()
        print(f"[{index + 1}/{len(runs)}] {cfg['engine']} / {cfg['model']} / {cfg['device']}", flush=True)
        result = {**run_isolated(cfg, directory), "id": job}
        results.append(result)
        export_results(output, results)  # Preserve completed jobs even if a later job fails.
        summary = result.get("summary", {})
        print(f"  {result['status']}: {result.get('reason', summary)}", flush=True)
    print(f"Results: {output}")
    return int(any(r["status"] == "error" for r in results) or not any(r["status"] == "ok" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
