import os
import tempfile

import soundfile as sf

TARGET_SR = 16000


def load(cfg):
    import torch
    from transformers import (AudioFlamingo3ForConditionalGeneration,
                              AutoProcessor)

    proc = AutoProcessor.from_pretrained(cfg["hf"])
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        cfg["hf"], device_map="auto", torch_dtype="auto")
    model.eval()
    return {"model": model, "proc": proc, "torch": torch}


def model_version(cfg, ctx):
    commit = getattr(ctx["model"].config, "_commit_hash", None)
    return f"{cfg['hf']}@{commit}" if commit else cfg["hf"]


def generate(ctx, *, prompt, audio, images, temperature, seed,
             max_new_tokens):
    import torch
    from transformers import set_seed

    model, proc = ctx["model"], ctx["proc"]
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        sf.write(tmp.name, audio[0], int(audio[1]), format="WAV")
        tmp.close()
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": tmp.name},
            ],
        }]
        inputs = proc.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True,
            return_dict=True,
        ).to(model.device)
        set_seed(int(seed))
        with torch.inference_mode():
            out = model.generate(
                **inputs, do_sample=True, temperature=float(temperature),
                max_new_tokens=int(max_new_tokens),
            )
        gen = out[:, inputs.input_ids.shape[1]:]
        return proc.batch_decode(gen, skip_special_tokens=True)[0]
    finally:
        os.unlink(tmp.name)
