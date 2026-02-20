"""
RunPod Serverless Handler for Qwen 3 (vLLM)
OpenAI-compatible chat/completion endpoint.
"""

import os
import runpod
from vllm import LLM, SamplingParams

# ─── Configuration ───────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-32B-AWQ")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
QUANTIZATION = os.environ.get("QUANTIZATION", "awq")

print(f"[Qwen Worker] Loading model: {MODEL_NAME}")
print(f"[Qwen Worker] Max context: {MAX_MODEL_LEN}, GPU util: {GPU_MEMORY_UTILIZATION}")
print(f"[Qwen Worker] Tensor parallel: {TENSOR_PARALLEL_SIZE}, Quantization: {QUANTIZATION}")

# ─── Load Model ──────────────────────────────────────────────────
llm = LLM(
    model=MODEL_NAME,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    quantization=QUANTIZATION if QUANTIZATION != "none" else None,
    trust_remote_code=True,
    dtype="half",
)

tokenizer = llm.get_tokenizer()


def build_prompt(messages):
    """Convert OpenAI-style messages to a chat prompt using the model's chat template."""
    if isinstance(messages, str):
        return messages

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: manual formatting
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


def handler(job):
    """Process a single inference request."""
    job_input = job["input"]

    # ─── Extract input ───────────────────────────────────────────
    # Support both "prompt" (raw string) and "messages" (OpenAI chat format)
    messages = job_input.get("messages")
    prompt = job_input.get("prompt", "")

    if messages:
        prompt = build_prompt(messages)
    elif not prompt:
        return {"error": "No 'prompt' or 'messages' provided"}

    # ─── Sampling parameters ─────────────────────────────────────
    temperature = job_input.get("temperature", 0.7)
    top_p = job_input.get("top_p", 0.9)
    top_k = job_input.get("top_k", 20)
    max_tokens = job_input.get("max_tokens", 2048)
    repetition_penalty = job_input.get("repetition_penalty", 1.05)
    stop = job_input.get("stop", None)
    stream = job_input.get("stream", False)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        stop=stop,
    )

    # ─── Generate ────────────────────────────────────────────────
    if stream:
        # Streaming mode
        def generate_stream():
            results = llm.generate(prompt, sampling_params, use_tqdm=False)
            for output in results:
                for token in output.outputs:
                    yield {"text": token.text}

        return generate_stream()
    else:
        # Standard mode
        results = llm.generate(prompt, sampling_params, use_tqdm=False)

        if not results or not results[0].outputs:
            return {"error": "No output generated"}

        output = results[0].outputs[0]

        return {
            "text": output.text,
            "usage": {
                "prompt_tokens": len(results[0].prompt_token_ids),
                "completion_tokens": len(output.token_ids),
                "total_tokens": len(results[0].prompt_token_ids) + len(output.token_ids),
            },
            "finish_reason": output.finish_reason,
        }


# ─── Start RunPod Worker ─────────────────────────────────────────
runpod.serverless.start({"handler": handler})
