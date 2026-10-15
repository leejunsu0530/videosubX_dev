"""
- 오디오의 whisper 엔진 목록: torch, huggingface(+optimum, mostly for cuda support), whisper.cpp(mostly for cpu), faster-whisper, 
openvino genai(for intel, expecially intel npu), optimum intel(for intel gpu)

whisperx의 최적화, 구조 확인.

urls:
- openai-whisper: https://github.com/openai/whisper
- huggingface(기억상으로, 둘이 구현이 다름):
    1. https://huggingface.co/openai/whisper-base
    2. https://huggingface.co/openai/whisper-large-v3
- whisper.cpp(=>pywhispercpp): https://github.com/absadiki/pywhispercpp
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- openvino-genai: 
    1. https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html?utm_source=chatgpt.com#whisper-inference-on-npu
    2. https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/speech-recognition/
- optimum-intel: 
    1. https://huggingface.co/docs/optimum-intel/openvino/inference
    2. https://huggingface.co/docs/optimum-intel/openvino/optimization
"""
