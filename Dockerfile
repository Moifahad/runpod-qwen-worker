FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install vLLM and RunPod SDK
RUN pip install --no-cache-dir \
    vllm>=0.6.0 \
    runpod>=1.7.0 \
    transformers>=4.45.0 \
    autoawq>=0.2.0

# Copy handler
COPY handler.py /app/handler.py

# Default environment (override in RunPod template)
ENV MODEL_NAME="Qwen/Qwen3-72B-AWQ"
ENV MAX_MODEL_LEN=32768
ENV GPU_MEMORY_UTILIZATION=0.90
ENV TENSOR_PARALLEL_SIZE=1
ENV QUANTIZATION=awq
# Set HF cache to network volume if mounted
ENV HF_HOME=/runpod-volume/huggingface

CMD ["python", "/app/handler.py"]
