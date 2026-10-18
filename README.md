# videosubX_dev
Make translated video and audio subtitle

Whisper 구현체 비교: [벤치마크 실행·설정 안내](test/WHISPER_BENCHMARK.md)

기능: 비디오와 오디오에 대한 번역된 자막 생성.
상세:
- 현재는 다운로드 지원 안함.
- 오디오 파이프라인: 오디오 파일 입력 -> UVR로 배경음 제거 -> silero-vad/pyannote로 VAD -> 여러 whisper 엔진으로 전사 -> whisperX로 음소 단위 타임스템프 / 화자분리 -> .srt / .ass subtitle
	- 오디오의 whisper 엔진 목록: torch, huggingface(+optimum, mostly for cuda support), whisper.cpp(mostly for cpu), faster-whisper, openvino genai(for intel, expecially intel npu), optimum intel(for intel gpu)
	- 각 전사 엔진에 여러 최적화 적용
- 비디오 파이프라인: 비디오 파일 입력 -> 이미지 추출 -> text detection OCR -> text recogntion OCR -> make into semi-inpainted(원본과 비슷한 배경색/글자색) or  translucent .ass subtitle
	- 다양한 OCR 엔진 지원
	- tracking으로 OCR 횟수 감소
	- 현재는 계획에 없지만, 필요해지면 hard imbedding & AI based inpainting 지원 가능
- 번역 파이프라인: huggingface 기반 번역 / 구글 번역 / deepL 번역 / google gemini 번역 등 지원
	- 일부 구현에 nidhaloff / deep-translator 리포지토리 사용으로 구글 번역 등 다양한 상용 번역기 지원
