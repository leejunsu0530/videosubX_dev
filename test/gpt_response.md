결론부터 말하면, **이 WhisperX는 VAD로 정한 구간을 실제 오디오에서 잘라 전사하고, 각 구간을 뒤쪽 zero-padding으로 30초에 맞춰 배치 처리합니다.** 최적화도 단순한 “배치 처리 + word timestamp off”보다 범위가 큽니다.

분석 기준은 로컬 클론의 `771b4a1` 커밋, 버전 `3.8.7rc1`입니다. 아래 설명은 기본 설정 기준입니다.

**1. VAD는 실제로 오디오를 어떻게 자르는가**

처리 흐름은 다음과 같습니다.

```text
전체 오디오: 16 kHz mono
  → VAD로 음성 구간 검출
  → 음성 구간들을 최대 약 30초 범위의 청크로 병합
  → 원본 파형에서 각 청크의 [start:end]를 슬라이스
  → 전사
```

기본 VAD는 Pyannote이며 Silero도 선택할 수 있습니다.

Pyannote는 프레임별 음성 점수를 얻은 다음, `Binarize`에서 시작 임계값 `0.5`, 종료 임계값 `0.363`으로 음성 구간을 만듭니다. 길게 이어지는 음성이 청크 길이를 넘으면, 현재 구간 **후반부에서 음성 점수가 가장 낮은 지점**을 찾아 나눕니다. 근거: [pyannote.py](/C:/Users/leeju/Projects/.Github_repo_example/whisperX/whisperx/vads/pyannote.py:108).

이후 `Vad.merge_chunks()`가 검출된 발화들을 합칩니다. 이때 기준은 **발화 길이의 합이 아니라 첫 발화 시작부터 마지막 발화 끝까지의 시간 범위**입니다. 근거: [vad.py](/C:/Users/leeju/Projects/.Github_repo_example/whisperX/whisperx/vads/vad.py:20).

예를 들어:

```text
검출된 발화: [2~5초] [8~12초] [40~45초]

병합된 청크: [2~12초] [40~45초]
전사 입력:   원본 audio[2초:12초], audio[40초:45초]
```

따라서:

- 첫 청크 안의 **5~8초 무음은 그대로 남습니다.**
- 청크 밖의 **12~40초 무음은 전사 입력에서 빠집니다.**
- 발화만 촘촘하게 이어 붙인 하나의 압축 오디오를 만드는 방식은 아닙니다.

실제로 전사 입력을 만드는 코드는 다음과 같습니다.

```python
f1 = int(seg['start'] * SAMPLE_RATE)
f2 = int(seg['end'] * SAMPLE_RATE)
yield {'inputs': audio[f1:f2]}
```

근거: [asr.py](/C:/Users/leeju/Projects/.Github_repo_example/whisperX/whisperx/asr.py:288).

원본 오디오를 수정하거나 잘라낸 파일을 저장하는 것은 아니고, **메모리에서 잘라낸 파형을 모델에 전달**합니다. 원본 시간 좌표도 유지되므로, 무음 제거에 따른 별도의 시간 복원 매핑이 필요 없습니다.

**2. 전사 입력의 패딩과 배치 구조**

각 청크는 Mel 변환 **전에 파형 뒤에 0을 채워 30초로 맞춥니다.**

```python
features = log_mel_spectrogram(
    audio,
    n_mels=...,
    padding=N_SAMPLES - audio.shape[0],
)
```

`N_SAMPLES = 30 × 16000 = 480000`이며, `log_mel_spectrogram()` 내부에서 다음을 실행합니다.

```python
audio = F.pad(audio, (0, padding))
```

근거: [전사 전처리](/C:/Users/leeju/Projects/.Github_repo_example/whisperX/whisperx/asr.py:191), [실제 패딩](/C:/Users/leeju/Projects/.Github_repo_example/whisperX/whisperx/audio.py:147).

즉, 입력은 다음처럼 됩니다.

```text
청크 A: 실제 10초 + zero-padding 20초
청크 B: 실제 27초 + zero-padding  3초
청크 C: 실제  5초 + zero-padding 25초

→ Mel: 각각 [n_mels, 3000]
→ stack: [batch_size, n_mels, 3000]
→ 배치 encoder
→ 배치 decoder generation
```

`n_mels`는 모델에 따라 80 또는 128입니다. 배치는 [torch.stack과 DataLoader](/C:/Users/leeju/Projects/.Github_repo_example/whisperX/whisperx/asr.py:236)로 구성하고, [generate_segment_batched()](/C:/Users/leeju/Projects/.Github_repo_example/whisperX/whisperx/asr.py:38)에서 encoder와 CTranslate2의 `generate()`를 호출합니다.

**배치 안에서 가장 긴 청크 길이에만 맞추는 동적 패딩은 아닙니다.** 짧은 청크도 기본적으로 30초 입력이 됩니다. `chunk_size`를 15초로 낮춰도 이 전처리의 패딩 목표는 여전히 30초입니다.

따라서 청크 병합에는 짧은 발화마다 30초 입력을 하나씩 만드는 낭비를 줄이는 효과도 있습니다.

**3. 바닐라 Whisper와 비교하면 무엇이 다른가**

| 항목 | OpenAI Whisper 기본 `transcribe()` | 이 WhisperX 기본 경로 |
|---|---|---|
| 오디오 분할 | 최대 30초 창을 순차 처리하고 출력 timestamp에 따라 다음 위치 결정 | VAD로 전사할 청크를 미리 확정 |
| 추론 엔진 | PyTorch | faster-whisper 기반 CTranslate2 |
| 청크 처리 | 기본 전사 루프는 순차 처리 | 여러 청크의 encoder·decoder 배치 추론 |
| timestamp 토큰 | 기본적으로 생성 | `without_timestamps=True` |
| word timestamp | 기본값부터 `False` | `False`, 이후 별도 align |
| 이전 청크 텍스트 | 기본적으로 다음 청크 prompt에 사용 | 기본적으로 사용하지 않음 |
| 실패 시 재시도 | temperature fallback 루프 존재 | 해당 배치 경로에는 fallback 루프 없음 |

바닐라 Whisper도 30초 입력에 맞추는 패딩을 사용합니다. 따라서 **패딩 자체가 WhisperX만의 최적화는 아닙니다.** 차이는 VAD로 청크를 미리 정하고, 청크 사이 의존성을 줄여 배치 추론할 수 있게 만든 구조에 있습니다. 바닐라 쪽 비교 근거: [OpenAI 전사 코드](https://github.com/openai/whisper/blob/main/whisper/transcribe.py), [오디오 처리 코드](https://github.com/openai/whisper/blob/main/whisper/audio.py).

특히 아래 두 설정은 구분해야 합니다.

```python
"without_timestamps": True,
"word_timestamps": False,
```

- `word_timestamps=False`: 단어별 시간 추정 후처리를 하지 않습니다.
- `without_timestamps=True`: **전사 decoder가 timestamp 토큰을 생성하지 않도록** 합니다.

WhisperX는 청크 경계를 VAD에서 이미 얻었으므로, 전사 중에는 텍스트 생성에 집중하고 정밀한 시간은 뒤에서 구합니다. 기본값은 [asr.py](/C:/Users/leeju/Projects/.Github_repo_example/whisperX/whisperx/asr.py:519)에 있습니다.

또한 `generate_segment_batched()`는 CTranslate2 `generate()`를 직접 호출합니다. 옵션 객체에는 `temperatures`, `compression_ratio_threshold`, `no_speech_threshold` 등이 있지만, **이 함수에서는 이 값으로 재시도하거나 무음 판정을 수행하지 않습니다.** 옵션이 존재한다는 것과 실제 배치 경로에서 사용된다는 것은 구분해야 합니다.

여기서 배치 generation 호출 한 번이 decoder의 forward 한 번을 의미하지는 않습니다. **텍스트 생성은 여전히 autoregressive이며, 기본 beam size도 5**입니다.

**4. align은 어떤 오디오를 받는가**

전사 직후 결과의 `start`, `end`는 Whisper가 예측한 문장 경계가 아니라 **병합된 VAD 청크 경계**입니다.

`align()`은 이 시간으로 **원본 오디오를 다시 슬라이스**합니다.

```python
waveform_segment = audio[:, f1:f2]
```

이후 별도의 음향 모델로 프레임별 CTC 출력을 얻고, 전사 텍스트를 trellis/backtracking으로 맞춰 문자·단어 시간을 계산합니다. **전사 때 추가했던 30초 zero-padding은 align에 전달되지 않습니다.** align 자체에는 아주 짧은 입력을 최소 400샘플로 맞추는 별도 패딩만 있습니다. 근거: [alignment.py](/C:/Users/leeju/Projects/.Github_repo_example/whisperX/whisperx/alignment.py:212).

이 구현에서 align은 청크별 순차 루프입니다. 전사 배치 최적화가 align까지 적용된 것은 아닙니다. CLI에서는 ASR·VAD 모델을 해제한 다음 align 모델을 로드해 GPU 메모리도 관리합니다. 근거: [transcribe.py](/C:/Users/leeju/Projects/.Github_repo_example/whisperX/whisperx/transcribe.py:163).

**5. 이 클론에 추가된 문맥 배치 옵션**

이 버전에는 `interleaved_context=True`도 있습니다. 기본값은 `False`입니다.

켜면 시간순 청크를 여러 연속 스트림으로 나누고, 각 배치에 스트림별 청크 하나씩을 넣습니다. 다음 배치에서는 같은 스트림의 이전 출력 토큰을 문맥으로 사용합니다. 첫 패스 후에는 스트림 시작 경계의 문맥을 보완하기 위해 첫 배치를 재전사합니다. 근거: [asr.py](/C:/Users/leeju/Projects/.Github_repo_example/whisperX/whisperx/asr.py:359).

따라서 이 옵션을 켠 경우에는 “모든 청크를 완전히 독립적으로 한 번씩 전사한다”는 설명이 맞지 않습니다.

질문하신 최적화의 핵심은 **VAD 기반 분할·병합 + 30초 고정 패딩 + timestamp 없는 배치 생성 + CTranslate2 실행 + 기본 문맥 의존성과 fallback 생략 + 별도 forced alignment**입니다. 이는 코드 경로 분석이며, 실제 속도나 정확도는 벤치마크하지 않았습니다.