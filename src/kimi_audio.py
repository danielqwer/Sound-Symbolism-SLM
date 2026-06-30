import os
import subprocess
import sys
import tempfile

import soundfile as sf

TARGET_SR = 16000

KIMI_REPO = "https://github.com/MoonshotAI/Kimi-Audio"
KIMI_COMMIT = "349251e"


def _ensure_kimia_infer():
    try:
        import kimia_infer  # noqa: F401
        return
    except ImportError:
        pass
    src = os.environ.get("KIMI_AUDIO_SRC") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "kimi_audio_src")
    if not os.path.isdir(os.path.join(src, "kimia_infer")):
        subprocess.run(["git", "clone", KIMI_REPO, src], check=True)
        subprocess.run(["git", "-C", src, "checkout", KIMI_COMMIT], check=True)
    sys.path.insert(0, src)


def load(cfg):
    _ensure_kimia_infer()
    from kimia_infer.api.kimia import KimiAudio

    model = KimiAudio(model_path=cfg["hf"], load_detokenizer=False)
    return {"model": model}


def model_version(cfg, ctx):
    return cfg["hf"]


def generate(ctx, *, prompt, audio, images, temperature, seed,
             max_new_tokens):
    model = ctx["model"]
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        sf.write(tmp.name, audio[0], int(audio[1]), format="WAV")
        tmp.close()
        messages = [
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "user", "message_type": "audio", "content": tmp.name},
        ]
        sampling_params = {"text_temperature": float(temperature)}
        _, text = model.generate(messages, **sampling_params,
                                 output_type="text",
                                 max_new_tokens=max(int(max_new_tokens), 16))
        return text
    finally:
        os.unlink(tmp.name)
