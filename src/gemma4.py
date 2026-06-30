import os
import tempfile

import soundfile as sf

TARGET_SR = 16000


def load(cfg):
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    proc = AutoProcessor.from_pretrained(cfg["hf"])
    model = AutoModelForMultimodalLM.from_pretrained(
        cfg["hf"], dtype="auto", device_map="auto")
    model.eval()
    return {"model": model, "proc": proc, "torch": torch}


def model_version(cfg, ctx):
    commit = getattr(ctx["model"].config, "_commit_hash", None)
    return f"{cfg['hf']}@{commit}" if commit else cfg["hf"]


def generate(ctx, *, prompt, audio, images, temperature, seed, max_new_tokens):
    import torch
    from transformers import set_seed
    model, proc = ctx["model"], ctx["proc"]
    tmp = []
    try:
        content = []
        if images:                                   # images before text
            if len(images) == 2:                     # exp3: label the two shapes
                content += [{"type": "text", "text": "Shape A:"},
                            {"type": "image", "image": images[0]},
                            {"type": "text", "text": "Shape B:"},
                            {"type": "image", "image": images[1]}]
            else:                                    # exp4: one shape
                content += [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        if audio is not None:                        # audio after text
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(f.name, audio[0], int(audio[1]), format="WAV")
            f.close(); tmp.append(f.name)
            content.append({"type": "audio", "audio": f.name})
        messages = [{"role": "user", "content": content}]
        set_seed(int(seed))
        inputs = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        plen = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max(int(max_new_tokens), 16),
                                 do_sample=True, temperature=float(temperature))
        text = proc.decode(out[0, plen:], skip_special_tokens=True)
        return text.strip()
    finally:
        for p in tmp:
            os.unlink(p)
