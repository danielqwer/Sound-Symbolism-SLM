import os
import numpy as np

TARGET_SR = 16000


def load(cfg):
    import torch
    from vllm import LLM
    from transformers import Qwen3OmniMoeProcessor

    tp = int(os.environ.get("VLLM_TP", "0")) or max(1, torch.cuda.device_count())
    llm = LLM(
        model=cfg["hf"],
        dtype="bfloat16",
        max_model_len=32768,
        max_num_seqs=16,
        limit_mm_per_prompt={"audio": 1, "image": 2, "video": 0},
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    proc = Qwen3OmniMoeProcessor.from_pretrained(cfg["hf"])
    return {"llm": llm, "proc": proc}


def model_version(cfg, ctx):
    return cfg["hf"]


def _build_content(prompt, audio, images):
    content = []
    if audio is not None:
        content.append({"type": "audio", "audio": ""})
    if images:
        if len(images) == 2:                       # exp3: A, B
            content += [
                {"type": "text", "text": "Shape A:"},
                {"type": "image", "image": ""},
                {"type": "text", "text": "Shape B:"},
                {"type": "image", "image": ""},
            ]
        else:                                       # exp4: single shape
            content.append({"type": "image", "image": ""})
    content.append({"type": "text", "text": prompt})
    return content


def generate(ctx, *, prompt, audio, images, temperature, seed,
             max_new_tokens):
    from vllm import SamplingParams

    llm, proc = ctx["llm"], ctx["proc"]
    conv = [{"role": "user", "content": _build_content(prompt, audio, images)}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True,
                                    tokenize=False)

    mm = {}
    if audio is not None:
        mm["audio"] = (np.asarray(audio[0], dtype=np.float32), audio[1])
    if images:
        mm["image"] = list(images)
    req = {"prompt": text}
    if mm:
        req["multi_modal_data"] = mm

    sp = SamplingParams(temperature=float(temperature),
                        max_tokens=int(max_new_tokens), seed=int(seed))
    out = llm.generate([req], sampling_params=sp, use_tqdm=False)
    return out[0].outputs[0].text
