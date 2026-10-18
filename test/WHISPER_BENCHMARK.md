# Whisper 구현체 벤치마크

프로젝트 루트에서 `python test/benchmark.py ...`로 실행합니다. 엔진별 새 프로세스에서 모델 하나를 로드하고, 각 오디오 전체를 워밍업한 다음 반복 추론합니다. 기존 `whisper_engine.py`의 링크 목록은 그대로 유지합니다. WhisperX는 목록의 설명대로 제외합니다.

## 설치

Python 3.11 이상, PATH에 등록된 FFmpeg가 필요합니다. mp3/mp4/wav/m4a 등 FFmpeg가 디코딩할 수 있는 입력의 **첫 번째 오디오 스트림**을 16 kHz mono float32로 읽어 모든 엔진에 동일한 배열을 전달합니다. 동영상의 영상 트랙은 처리하지 않습니다.

```powershell
python -m pip install -r test/requirements-benchmark.txt
# 사용할 엔진만 설치하세요.
python -m pip install openai-whisper
python -m pip install torch transformers accelerate
python -m pip install pywhispercpp
python -m pip install faster-whisper
python -m pip install "optimum-intel[openvino]" openvino-genai
```

PyTorch는 장치에 맞는 CPU/CUDA/XPU 빌드를 사용합니다. 엔진들의 의존성 버전이 충돌하면 가상환경을 나누고 JSON의 각 run에 `"python": "C:/path/to/env/Scripts/python.exe"`를 지정하세요. 컨트롤러에는 추가 패키지가 필요 없으며 **각 worker 환경**에 공통 의존성과 해당 엔진을 설치합니다. 기존 프로젝트의 `ovgenai` extra에는 NPU 문서의 버전 조합이 있습니다. 벤치마크를 위해 프로젝트 전체 의존성을 설치할 필요는 없습니다.

## 바로 실행

```powershell
python test/benchmark.py --list-engines
python test/benchmark.py --engine openai hf-pipeline hf-direct whispercpp faster-whisper --model base --audio "D:/audio/test.m4a" --language ko --device cpu --reference "D:/audio/test.txt" --warmup 1 --repeats 3
python test/benchmark.py --engine faster-whisper --model large-v3 --audio "D:/audio/test.mp4" --device cuda:0 --compute-type float16
```

`--reference`는 자막 타임코드가 없는 UTF-8 정답 전사문 파일입니다. 생략하면 시간/메모리만 측정하고 정확도는 null입니다. WER/CER은 오류율이며 낮을수록 좋습니다. 삽입 오류가 많으면 1(100%)을 넘을 수 있으므로 `1-WER`를 정확도로 해석하지 마세요.

엔진을 생략하면 7개 경로를 모두 시도합니다. HF 엔진은 `base`를 `openai/whisper-base`로 확장합니다. 다른 HF 모델 ID와 로컬 모델 경로도 가능합니다. OpenAI/faster-whisper/whisper.cpp는 각 구현체의 모델명 또는 모델 경로를 사용합니다. 예를 들어 whisper.cpp 양자화 모델은 GGML 파일을, faster-whisper는 CTranslate2 폴더를 지정합니다. 동일한 이름이라도 실제 가중치/양자화가 다를 수 있으므로 비교 시 원본 모델 버전을 맞추세요.

| 엔진 | 장치 설정 | 비고 |
| --- | --- | --- |
| openai | cpu, cuda:0 | 공식 Whisper transcribe |
| hf-pipeline | cpu, cuda:0, xpu:0 | Transformers ASR pipeline |
| hf-direct | cpu, cuda:0, xpu:0 | processor + generate |
| whispercpp | cpu | 일반 pip 빌드, GPU 사용을 명시적으로 끔 |
| faster-whisper | cpu, cuda:0 | CTranslate2, generator 소비까지 측정 |
| openvino-genai | cpu, gpu.0, npu | OpenVINO IR 모델 필요 |
| optimum-intel | cpu, gpu.0 | OVModelForSpeechSeq2Seq + generate |

OpenVINO `GPU.0`가 내장 GPU인지 외장 GPU인지는 시스템에 따라 다릅니다. `python -c "import openvino as ov; c=ov.Core(); print([(d,c.get_property(d,'FULL_DEVICE_NAME')) for d in c.available_devices])"`로 실제 장치 이름을 확인하세요. `xpu:0`는 PyTorch XPU가 지원하는 Intel GPU 경로입니다. `gpu`를 CUDA나 XPU로 자동 매핑하지 않습니다. 지원 여부는 장치/드라이버/모델/패키지 빌드에 따라 달라지며 컴파일 또는 추론 실패는 오류로 기록합니다. 지원하지 않는 엔진 조합·없는 장치·없는 패키지는 `skipped`로 기록하고 다음 작업을 진행합니다. CPU로 자동 전환하지 않습니다.

## OpenVINO 모델 준비

변환/양자화는 벤치마크 전에 수행합니다. HF ID를 OpenVINO 엔진에 바로 전달하지 않습니다.

```powershell
optimum-cli export openvino --model openai/whisper-base --weight-format fp16 test/models/whisper-base-fp16
optimum-cli export openvino --model openai/whisper-base --weight-format int8 test/models/whisper-base-int8
python test/benchmark.py --engine openvino-genai --model test/models/whisper-base-fp16 --audio "D:/audio/test.wav" --device cpu gpu.0 npu --language ko
python test/benchmark.py --engine optimum-intel --model test/models/whisper-base-fp16 --audio "D:/audio/test.wav" --device gpu.0 --language ko
```

최신 GenAI는 `ASRPipeline`, 일부 버전은 `WhisperPipeline`을 제공하므로 둘 다 처리합니다. NPU는 OpenVINO GenAI 경로로 실행합니다. OpenVINO 2025.1 이상에서는 Whisper 변환 시 `--disable-stateful`이 필수가 아닙니다. INT8 가중치 압축과 INT8 활성화 양자화는 서로 다른 실험입니다. 후자는 공식 최적화 문서대로 데이터셋으로 보정해 별도 폴더에 내보낸 후 모델 경로를 지정하세요.

## 여러 모델·오디오·장치 조합

`test/whisper_benchmark.example.json`을 복사해 오디오/정답/모델 경로를 바꾸세요. JSON 파일의 상대 경로는 **설정 파일 위치**를 기준으로 해석합니다. `defaults`보다 각 `runs` 항목이 우선합니다. `samples`도 run별로 바꿀 수 있습니다. `load_options`/`decode_options`는 해당 API의 추가 키워드 인자이며 기본 딕셔너리를 통째로 대체합니다. `decode_options`는 기본 디코딩 값을 덮어쓰므로 비교 설정을 결과 JSON에서 확인하세요. 장치/정밀도/스레드는 전용 상위 설정으로 지정합니다.

```powershell
python test/benchmark.py --config test/my-benchmark.json --dry-run
python test/benchmark.py --config test/my-benchmark.json --output benchmark-results/my-comparison
```

설정 파일 사용 시 CLI의 개별 추론 옵션은 적용하지 않습니다. `--output`과 `--dry-run`은 사용할 수 있습니다. `--dry-run`은 경로와 설정만 검사하며 장치/패키지/모델 실행을 검사하지 않습니다.

많은 오디오는 JSONL manifest로도 지정할 수 있습니다. 줄마다 아래 형태로 기록하고 `--manifest test/corpus.jsonl`로 실행합니다. 경로는 manifest 파일 위치 기준입니다.

```jsonl
{"audio":"audio/a.mp3","reference":"여기에 첫 번째 정답 전사문"}
{"audio":"audio/b.mp4","reference_file":"audio/b.txt"}
```

기본 디코딩은 transcription, beam 1, 지원 경로에서 temperature 0, faster-whisper VAD 비활성화입니다. 자동 언어 감지 대신 `--language ko` 등으로 고정하면 비교 조건을 맞추기 좋습니다. HF 두 경로 및 Optimum은 전체 입력과 attention mask/타임스탬프를 이용하는 순차 long-form 처리를 사용합니다. 30초 이후를 잘라 버리지 않습니다. 각 엔진의 무음 처리·문맥·fallback·timestamp 정책까지 동일하지는 않으므로 이 도구는 **각 API를 통한 전체 전사 성능**을 비교합니다. `beam_size`, 정밀도, 양자화, 추가 옵션을 맞추고 해석하세요. Torch 기본 dtype와 CTranslate2 기본 compute_type은 모두 float32입니다. whisper.cpp와 OpenVINO 정밀도는 모델 파일/런타임 설정에 의해 결정됩니다.

## 측정값과 결과

기본 결과 위치는 `benchmark-results/<실행시각>/`입니다. 기존 비어 있지 않은 폴더는 덮어쓰지 않습니다.

- `results.json`: 환경/패키지 버전/설정/실제 모델 ID, 각 전사문과 정답, 측정값, 실패 사유.
- `summary.csv`: 실행 조합별 로딩 시간, 평균/중앙값/표준편차, RTF, corpus WER/CER.
- `transcriptions.csv`: 샘플·반복별 시간/메모리/전사문/정확도.
- `<번호>-<엔진>/worker.log`: 엔진 로그와 네이티브 오류. request/result JSON도 보존합니다.

`load_seconds`는 라이브러리 import + 모델 로딩/다운로드/즉시 컴파일 시간이며 Python 프로세스 시작은 제외합니다. 사전에 모델을 받아 두어야 로딩 시간을 공정하게 비교할 수 있습니다. `wall_seconds`는 프로세스 시작부터 모든 샘플·워밍업·반복·평가·종료까지입니다. 디스크/컴파일 캐시를 지우지 않으며 첫 실행과 후속 실행의 차이가 남을 수 있습니다.

`seconds`는 입력 배열의 특징 추출부터 텍스트 생성/디코딩·장치 동기화까지입니다. FFmpeg 디코딩은 별도 `audio_preparation` 시간이며 추론 시간/RTF에서 제외합니다. CPU에서 수행되는 전처리·후처리도 포함됩니다. 기본 워밍업은 **샘플마다 전체 오디오 1회**이므로 긴 파일에서는 비용이 큽니다. `--warmup 0`으로 첫 추론 성능을 비교할 수도 있습니다. `rtf = 추론초 / 오디오초`이며 1보다 작으면 실시간보다 빠릅니다. corpus RTF는 반복을 포함한 전체 추론초/전체 오디오초입니다. 길이가 다른 오디오를 섞은 평균 latency보다 샘플별 시간과 corpus RTF를 함께 보세요.

RAM은 psutil로 worker와 살아 있는 자식 프로세스의 RSS 합계를 기본 20ms마다 샘플링합니다. 로딩/오디오 준비/워밍업/추론별 peak와 추론 직전 RSS, 추론 중 증가량을 구분합니다. peak는 근사치이며 짧은 순간 최대치를 놓칠 수 있습니다. 공유 페이지는 중복 집계될 수 있고, RSS는 시스템 전체 RAM이나 GPU/NPU 전용 메모리가 아닙니다. `--sample-interval`로 간격을 조절할 수 있습니다.

Torch CUDA 경로만 allocator의 peak allocated/reserved MiB를 추가 기록합니다. 모델 메모리도 포함되며 CUDA 전체 프로세스 메모리는 아닙니다. faster-whisper CUDA와 Intel GPU/NPU의 전용 메모리는 측정하지 않아 해당 값이 null입니다. 통합 메모리 시스템에서도 RSS를 장치 메모리로 해석하지 마세요.

`normalization=basic`은 Unicode NFKC, casefold, Unicode 구두점 제거, 연속 공백 정리를 적용합니다. WER는 공백 단어 기준이며, CER는 공백을 제외한 Unicode 코드포인트 기준입니다. `none`은 문자 정규화를 생략하되 WER의 공백 분리와 CER의 공백 제외는 유지합니다. 한국어는 띄어쓰기에 민감한 WER와 CER를 함께 확인하세요. 공식 데이터셋 고유의 normalizer나 Whisper English normalizer는 사용하지 않습니다. 오류율은 Levenshtein 편집거리/정답 길이입니다. 빈 정답에 삽입만 있는 경우 오류 개수는 기록하고 분모가 0인 오류율은 null로 남깁니다. corpus 오류율은 첫 번째 측정 반복의 정답 있는 샘플들을 합산해 계산하며 반복 수로 가중하지 않습니다.

오류가 있거나 성공한 조합이 하나도 없으면 종료 코드 1입니다. 일부 조합이 skipped여도 성공 조합이 있고 오류가 없으면 0입니다. 작업마다 기본 3600초 제한(`--timeout`)을 적용하며 완료한 조합은 즉시 저장합니다.

## 자체 검증

```powershell
python -m unittest discover -s test -p "test_whisper_benchmark.py" -v
```

외부 모델을 다운로드하지 않고 정확도 계산, 가중 집계, 설정 경로, 장치 거부, 어댑터 호출, worker 실패 격리를 검사합니다. 실제 모델 성능은 해당 장치와 패키지가 설치된 환경에서 별도로 실행해야 합니다.

공통 의존성의 RapidFuzz로 긴 전사문의 편집거리를 빠르게 계산합니다. RapidFuzz가 없으면 동일한 정의의 순수 Python 구현으로 계산하며 긴 파일에서는 평가 시간이 크게 늘 수 있습니다. 평가 시간은 추론 시간에 포함하지 않습니다.

## 공식 API 근거

- [OpenAI Whisper](https://github.com/openai/whisper)
- [HF base 직접 호출 예제](https://huggingface.co/openai/whisper-base), [large-v3 pipeline / 직접 호출 / long-form](https://huggingface.co/openai/whisper-large-v3)
- [pywhispercpp](https://github.com/absadiki/pywhispercpp)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [GenAI ASR](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/speech-recognition/), [NPU Whisper](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html#whisper-inference-on-npu)
- [Optimum Intel inference](https://huggingface.co/docs/optimum-intel/openvino/inference), [optimization](https://huggingface.co/docs/optimum-intel/openvino/optimization)
