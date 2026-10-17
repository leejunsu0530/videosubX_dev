"""
- 오디오의 whisper 엔진 목록

urls:
- ~whisperX: https://github.com/m-bain/whisperX~
    - whisperx는 faster-whisper를 사용하니까, 모델 자체의 속도를 비교하는 이 테스트에선 배제
- openai-whisper: https://github.com/openai/whisper
    - 
- huggingface(+optimum, mostly for cuda support):
    - 한 쪽은 파이프라인으로, 다른 쪽은 수동 호출로 된 예제 코드. 각각을 따로 테스트할 것.
    - optimum은 아래에서 따로 테스트하고 여기는 예제 코드대로 바닐라 hf만.
    1. https://huggingface.co/openai/whisper-base
    2. https://huggingface.co/openai/whisper-large-v3
- whisper.cpp(mostly for cpu): https://github.com/absadiki/pywhispercpp
    - whisper.cpp 래퍼인 pywhispercpp로 사용. cpu를 사용하니까 일반 빌드로 해서(pip install pywhispercpp) 테스트.
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- openvino-genai(for intel npu/gpu, expecially intel npu): 
    1. https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html?utm_source=chatgpt.com#whisper-inference-on-npu
    2. https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/speech-recognition/
- optimum-intel(for intel gpu): 
    1. https://huggingface.co/docs/optimum-intel/openvino/inference
    2. https://huggingface.co/docs/optimum-intel/openvino/optimization
"""
