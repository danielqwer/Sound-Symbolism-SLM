import os
import numpy as np

TARGET_SR = 16000


def load(cfg):
    import torch
    from vllm import LLM
    from transformers import AutoTokenizer

    tp = int(os.environ.get("VLLM_TP", "0")) or max(1, torch.cuda.device_count())
    llm = LLM(
        model=cfg["hf"],
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        limit_mm_per_prompt={"audio": 1, "image": 2},
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.9,
    )
    tok = AutoTokenizer.from_pretrained(cfg["hf"], trust_remote_code=True)
    return {"llm": llm, "tok": tok}


def model_version(cfg, ctx):
    return cfg["hf"]


def _user_text(prompt, audio, images):
    parts = []
    if images:
        if len(images) == 2:                         # exp3: label the two shapes
            parts += ["Shape A:", "(<image>./</image>)",
                      "Shape B:", "(<image>./</image>)"]
        else:                                        # exp4: single image
            parts += ["(<image>./</image>)"] * len(images)
    if audio is not None:
        parts.append("(<audio>./</audio>)")
    parts.append(prompt)
    return "\n".join(parts)


def generate(ctx, *, prompt, audio, images, temperature, seed,
             max_new_tokens):
    from vllm import SamplingParams

    llm, tok = ctx["llm"], ctx["tok"]
    msgs = [{"role": "user", "content": _user_text(prompt, audio, images)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)

    mm = {}
    if audio is not None:
        mm["audio"] = [(np.asarray(audio[0], dtype=np.float32), audio[1])]
    if images:
        mm["image"] = list(images)
    req = {"prompt": text}
    if mm:
        req["multi_modal_data"] = mm

    sp = SamplingParams(temperature=float(temperature),
                        max_tokens=int(max_new_tokens), seed=int(seed))
    out = llm.generate([req], sampling_params=sp, use_tqdm=False)
    return out[0].outputs[0].text
