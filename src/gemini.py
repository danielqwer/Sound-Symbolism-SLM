import io
import os

import numpy as np
import soundfile as sf


def load(cfg):
    from google import genai

    key = os.environ.get(cfg.get("api_key_env", "GEMINI_API_KEY"))
    return {"client": genai.Client(api_key=key), "model": cfg["api_model"]}


def model_version(cfg, ctx):
    # API model string; the exact served version is not exposed pre-call.
    return cfg["api_model"]


def _wav_bytes(audio):
    arr, sr = audio
    buf = io.BytesIO()
    sf.write(buf, np.asarray(arr, dtype=np.float32), int(sr), format="WAV")
    return buf.getvalue()


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate(ctx, *, prompt, audio, images, temperature, seed,
             max_new_tokens):
    from google.genai import types

    parts = [types.Part.from_text(text=prompt)]
    if audio is not None:
        parts.append(types.Part.from_bytes(
            data=_wav_bytes(audio), mime_type="audio/wav"))
    if images:
        if len(images) == 2:                     # exp3: A, B
            for label, img in zip(("A:", "B:"), images):
                parts.append(types.Part.from_text(text=label))
                parts.append(types.Part.from_bytes(
                    data=_png_bytes(img), mime_type="image/png"))
        else:                                    # exp4: single shape
            for img in images:
                parts.append(types.Part.from_bytes(
                    data=_png_bytes(img), mime_type="image/png"))

    kw = dict(temperature=temperature, seed=int(seed),
              max_output_tokens=max(int(max_new_tokens), 64),
              thinking_config=types.ThinkingConfig(thinking_budget=0))
    try:
        cfg = types.GenerateContentConfig(**kw)
    except TypeError:                       # SDK build without thinking_config
        kw.pop("thinking_config", None)
        cfg = types.GenerateContentConfig(**kw)

    resp = ctx["client"].models.generate_content(
        model=ctx["model"],
        contents=[types.Content(role="user", parts=parts)],
        config=cfg,
    )
    return resp.text or ""
