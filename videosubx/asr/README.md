# 교체 가능한 Whisper 전사

`videosubx.asr`는 WhisperX의 VAD를 재사용하고 전사 엔진만 교체한다.
기존 `test/benchmark.py`의 원본 엔진 비교와 독립적인 제품용 모듈이다.

공통 경로:

```text
원본 16 kHz mono → VAD → 최대 30초 범위로 병합 → 원본 파형 슬라이스
  → 청크 뒤를 0으로 채워 30초 → timestamp 없는 전사 → 원본 시간의 segments
  → 선택적으로 WhisperX forced alignment (원본 오디오 사용)
```

병합된 청크 내부 무음은 유지된다. 청크 밖 무음은 전사 입력에서 제외된다.
`chunk_size=15`로 낮춰도 encoder 입력 패딩은 30초다.
VAD와 alignment는 전사 엔진과 다른 장치에서 실행할 수 있다.

## Python 사용

저장소 루트에서 실행하거나 이 프로젝트를 설치한 환경에서 사용한다.
`ffmpeg`가 PATH에 있어야 한다. 사용할 엔진의 패키지만 해당 환경에 설치한다.
기존 프로젝트 전체 의존성 설치가 부담되면 각 엔진용 가상환경에서 저장소 루트를
작업 디렉터리로 두고 이 모듈을 직접 사용해도 된다.

```python
from videosubx.asr import (
    TranscriptionPipeline, WhisperXVAD, load_backend, load_audio, align_result,
)

audio = load_audio("input.wav")
vad = WhisperXVAD("silero", device="cpu", chunk_size=30)
backend = load_backend("whispercpp", "base", device="cpu", beam_size=5)
pipe = TranscriptionPipeline(backend, vad)
result = pipe.transcribe(audio, language="ko", batch_size=1)
# result = {"segments": [{"start": ..., "end": ..., "text": ...}], "language": "ko"}

# 필요하면 ASR/VAD 객체를 먼저 해제한 후 실행한다.
result = align_result(result, audio, device="cpu")
```

엔진 교체는 `load_backend()` 한 줄만 바꾼다:

```python
backend = load_backend("huggingface", "openai/whisper-base", device="cuda:0", dtype="float16")
backend = load_backend("faster-whisper", "base", device="cuda:0", compute_type="float16")
backend = load_backend("openai", "base", device="cpu")
backend = load_backend("optimum-intel", "./models/whisper-base-ov", device="gpu")
backend = load_backend("optimum-onnx", "./models/whisper-base-onnx", device="cpu")
```

Optimum은 해당 런타임으로 미리 export한 모델과 processor 파일을 준비한다.
OpenVINO 장치는 `cpu`, `gpu`, `gpu.0` 등을 받는다. ONNX는 `cpu`, `cuda:0` 등을 받으며
CPU/GPU에 맞는 ONNX Runtime provider 패키지가 필요하다.
로드 세부 옵션은 `model_options={"cache_dir": ..., "local_files_only": True}`처럼 넘길 수 있다.
이 옵션은 엔진의 모델 생성자로 전달되므로 해당 생성자의 지원 여부를 확인한다.
Transformers processor도 별도로 로드하므로 processor 옵션을 제어하려면 아래처럼 주입한다.

## 지원 범위와 차이

| 이름 | 패키지 | 전사 경로 | 청크 배치 |
| --- | --- | --- | --- |
| `openai` | `openai-whisper`, `torch` | 배치 Mel → `whisper.decode()` | 지원 |
| `huggingface` | `transformers`, `torch` | processor → `generate()` | 지원 |
| `faster-whisper` | `faster-whisper` | features → encoder → CTranslate2 `generate()` | 지원 |
| `whispercpp` | `pywhispercpp` | `Model.transcribe()` | 순차 처리 |
| `optimum-intel` | `optimum-intel[openvino]`, `torch` | processor → OV `generate()` | API상 지원; export/장치별 확인 필요 |
| `optimum-onnx` | `optimum-onnx[onnxruntime]`, `torch` | processor → ORT `generate()` | API상 지원; export별 확인 필요 |

기본 VAD/align은 `whisperx`와 그 의존성이 필요하다. VAD를 직접 넣거나
`speech_regions`를 제공하고 align을 사용하지 않으면 WhisperX를 import하지 않는다.
WhisperX 자체 VAD 모듈은 Pyannote 등 다른 의존성도 import할 수 있다.

- 언어는 `ko`, `en` 등으로 명시한다. 청크마다 언어 감지를 반복하지 않는다.
- 모든 엔진에서 청크 간 텍스트 문맥은 독립적이다. `initial_prompt`는 각 청크에 같은 값을 적용한다.
- `interleaved_context`, 자동 언어 감지, 번역 모드는 이번 인터페이스에 포함하지 않았다.
- 높은 수준의 long-form `transcribe()`를 다시 실행하는 대신 short-form 경로를 사용한다.
  pywhispercpp는 공개 `transcribe()` API 안에서 처리하므로 자체 no-speech 등 내부 정책은 남는다.
- Python 측 temperature fallback 재시도는 하지 않는다. C++ 내부 재시도 정책은 런타임에 따른다.
- `dtype`는 PyTorch 엔진, `compute_type`은 faster-whisper에 적용한다. C++/Optimum 정밀도는 모델 파일과 export 설정에서 결정한다.
- 엔진별 Mel 계산, beam search, 숫자 정밀도와 내부 구현은 달라 텍스트·속도의 완전한 일치는 보장하지 않는다.
- pywhispercpp는 `no_context`, `no_timestamps`, `single_segment`를 설정한다.
  `batch_size > 1`은 순차 처리 경고를 표시한다. 실제 GPU 배치를 제공하는 것처럼 취급하지 않는다.
  내장 로더는 CPU용이며 GPU로 빌드한 모델은 `WhisperCppBackend(model)`로 직접 주입할 수 있다.
- Optimum 모델은 고정 batch=1 export라면 `batch_size=1`로 실행하거나 동적 배치로 다시 export한다.
  실패를 감추는 자동 순차 재시도는 없다.
- 공통 파형 패딩은 동일하지만 Mel 전처리는 각 엔진의 네이티브 구현을 사용한다.
  HF/Optimum processor에는 원래 길이의 파형과 30초 패딩 목표를 주어 attention mask를 함께 만든다.
- `avg_logprob`는 OpenAI/faster-whisper에서만 반환한다. 엔진 간 확률값 비교를 보장하지 않는다.
- align은 정확한 단어 시간을 붙이는 별도 작업이며 이 모듈에서 배치화하지 않았다.

## VAD를 캐시하거나 바꾸기

엔진 비교 시 같은 VAD 결과를 한 번 만들어 재사용할 수 있다.
시간은 원본 파일 기준 초 단위이며, 실제 샘플링 레이트는 16000이어야 한다.

```python
regions = list(vad(audio))
pipe = TranscriptionPipeline(backend)
result = pipe.transcribe(audio, language="ko", speech_regions=regions, batch_size=8)
```

직접 검출한 결과도 `[{"start": 2.0, "end": 5.0}, ...]` 형태로 받는다.
정렬되지 않은 구간·겹치는 구간·30초를 넘는 단일 발화는 오류를 낸다.
긴 발화 분할은 VAD가 수행해야 하며, 기본 WhisperX Pyannote VAD는 min-cut을 사용한다.
`speech_regions=[]`는 무음이다. VAD와 구간을 둘 다 생략하면 오류를 낸다.

## 새 엔진 추가 / 로드한 모델 재사용

`Backend` 프로토콜의 `name`, `supports_batch`, `decode()`만 구현한다.
`decode()`는 입력 순서대로 청크당 `Decoded` 하나를 반환해야 한다.
청크 `audio`는 480000샘플 float32, `valid_samples`는 패딩 전 길이다.
시간 좌표는 공통 파이프라인이 관리하므로 어댑터가 다시 계산하지 않는다.

```python
from videosubx.asr import Decoded, TranscriptionPipeline, register_backend

class MyBackend:
    name = "custom"
    supports_batch = True

    def __init__(self, model):
        self.model = model

    def decode(self, chunks, *, language):
        texts = self.model.generate_batch([c.audio for c in chunks], language=language)
        return [Decoded(text) for text in texts]

pipe = TranscriptionPipeline(MyBackend(my_model), vad)
# 또는 register_backend("custom", factory)를 사용해 load_backend()에서 생성한다.
# factory(model_id, *, device, config, dtype, compute_type, model_options)
```

이미 로드한 HF/Optimum 모델은 `TransformersBackend(model, processor, ...)`,
OpenAI는 `OpenAIBackend(model)`, faster-whisper는 `FasterWhisperBackend(model)`,
C++는 `WhisperCppBackend(model)`로 감쌀 수 있다.
C++의 beam sampling strategy는 모델 생성 시 선택해야 한다. beam_size>1인
DecodeConfig를 주입할 때 모델도 `params_sampling_strategy=1`로 생성한다.

## CLI

```powershell
python -m videosubx.asr input.wav --engine whispercpp --model base --language ko --batch-size 1
python -m videosubx.asr input.wav --engine huggingface --model openai/whisper-base --language ko --device cuda:0 --dtype float16 --batch-size 8 --align --align-device cuda:0 --output result.json
python -m videosubx.asr input.wav --engine optimum-intel --model ./models/whisper-base-ov --language ko --device gpu --batch-size 1 --speech-regions regions.json
```

## 검증

```powershell
python -m unittest discover -s test -p test_asr_pipeline.py -v
```

이 테스트는 실제 파형 슬라이싱·패딩·배치·시간 보존을 검사하고 외부 엔진 API는
대역으로 검사한다. 모델 실행 패키지와 가중치가 없는 환경에서 검증했으므로,
실제 모델 추론·Optimum export별 batch 지원·속도·WER는 별도 검증해야 한다.
실행 패키지의 API가 바뀌면 어댑터의 호환성을 다시 확인해야 한다.

공식 API: [OpenAI decode](https://github.com/openai/whisper/blob/main/whisper/decoding.py),
[Transformers Whisper](https://huggingface.co/docs/transformers/model_doc/whisper),
[pywhispercpp](https://absadiki.github.io/pywhispercpp/),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[Optimum Intel](https://huggingface.co/docs/optimum-intel/openvino/reference),
[Optimum ONNX](https://huggingface.co/docs/optimum-onnx/onnxruntime/package_reference/modeling_ort).
