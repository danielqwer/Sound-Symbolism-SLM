import os
import subprocess
import sys
import tempfile

import soundfile as sf

TARGET_SR = 16000

STEP_REPO = "https://github.com/stepfun-ai/Step-Audio2"
STEP_COMMIT = "76e272b"


def _ensure_step_src():
    repo = os.environ.get("STEP_AUDIO2_REPO")
    if repo:
        if repo not in sys.path:
            sys.path.insert(0, repo)
        return
    try:
        import stepaudio2  # noqa: F401
        return
    except ImportError:
        pass
    src = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "step_audio2_src")
    if not os.path.isfile(os.path.join(src, "stepaudio2.py")):
        subprocess.run(["git", "clone", STEP_REPO, src], check=True)
        subprocess.run(["git", "-C", src, "checkout", STEP_COMMIT], check=True)
    sys.path.insert(0, src)


def load(cfg):
    _ensure_step_src()
    from stepaudio2 import StepAudio2

    model = StepAudio2(cfg["hf"])
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
        # system=instruction, human=audio, empty assistant; __call__ -> (ids, text, audio_tokens)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "human", "content": [{"type": "audio", "audio": tmp.name}]},
            {"role": "assistant", "content": None},
        ]
        out = model(messages, max_new_tokens=max(int(max_new_tokens), 16),
                    temperature=float(temperature), do_sample=True)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            return out[1]                      # output_text
        return out if isinstance(out, str) else str(out)
    finally:
        os.unlink(tmp.name)
