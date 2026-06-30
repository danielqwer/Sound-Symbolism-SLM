import base64
import io
import os

import numpy as np
import soundfile as sf

TARGET_SR = 16000   # keep the uploaded WAV small; the API resamples as needed


def load(cfg):
    from openai import OpenAI

    key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"))
    return {"client": OpenAI(api_key=key), "model": cfg["api_model"]}


def model_version(cfg, ctx):
    # API model string; the exact served snapshot is not exposed pre-call.
    return cfg["api_model"]


def _wav_b64(audio):
    arr, sr = audio
    buf = io.BytesIO()
    sf.write(buf, np.asarray(arr, dtype=np.float32), int(sr), format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate(ctx, *, prompt, audio, images, temperature, seed,
             max_new_tokens):
    if audio is None:
        raise ValueError("gpt-audio-1.5 is audio-only; got no audio")

    content = [
        {"type": "text", "text": prompt},
        {"type": "input_audio",
         "input_audio": {"data": _wav_b64(audio), "format": "wav"}},
    ]
    # speech model: must request audio output and read the transcript (text-only refuses)
    resp = ctx["client"].chat.completions.create(
        model=ctx["model"],
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        seed=int(seed),
        max_completion_tokens=max(int(max_new_tokens), 256),
    )
    m = resp.choices[0].message
    return (m.content or (getattr(m, "audio", None) and m.audio.transcript) or "")
