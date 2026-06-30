import io
import os

import numpy as np

# Required output schema (per project spec). Order is the column order.
OUTPUT_COLUMNS = [
    "trial_id", "experiment", "model", "model_version", "stimulus_id",
    "sample_idx", "raw_response", "parsed_value", "parse_ok",
    # recommended extras (handy for analysis / debug)
    "scale", "prompt_lang", "option_order", "latency_ms", "timestamp", "run_id",
]


def _local_trials_path(exp):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, os.pardir, "data", "trials", f"{exp}.csv")


def load_trials(exp):
    import pandas as pd

    df = pd.read_csv(_local_trials_path(exp))
    df = df.astype(object).where(pd.notna(df), None)   # NaN (empty scale/order) -> None
    return df.to_dict("records")


def load_stimuli(exp):
    raise RuntimeError("Stimuli are not bundled with this repo yet — "
                       "the dataset will be released soon.")


# Media decoding
def decode_audio(media, target_sr=None):
    """media: {'bytes': ...}. Returns (np.float32 mono array, sr)."""
    import soundfile as sf

    arr, sr = sf.read(io.BytesIO(media["bytes"]), dtype="float32")
    if arr.ndim > 1:                      # stereo -> mono
        arr = arr.mean(axis=1)
    arr = np.asarray(arr, dtype=np.float32)
    if target_sr and sr != target_sr:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return arr, sr


def decode_image(media):
    """media: {'bytes': ...}. Returns a PIL.Image (RGB)."""
    from PIL import Image

    return Image.open(io.BytesIO(media["bytes"])).convert("RGB")


# Per-trial media -> {"audio": (array, sr) | None, "images": [A, B] | None}.
# For exp3 the first image is option A, second is B (per option_order).
def build_media(exp, trial, stim_by_id, target_sr=None):
    sid = trial["stimulus_id"]
    s = stim_by_id[sid]
    if exp in ("exp1", "exp2"):
        return {"audio": decode_audio(s["audio"], target_sr), "images": None}
    if exp == "exp3":
        audio = decode_audio(s["audio"], target_sr)
        rounded = decode_image(s["rounded_shape"])
        spiky = decode_image(s["spiky_shape"])
        # option_order says which shape is presented as A (the first image).
        if trial["option_order"] == "rounded_first":
            images = [rounded, spiky]           # A=rounded, B=spiky
        elif trial["option_order"] == "spiky_first":
            images = [spiky, rounded]           # A=spiky, B=rounded
        else:
            raise ValueError(f"bad option_order: {trial['option_order']!r}")
        return {"audio": audio, "images": images}
    if exp == "exp4":
        return {"audio": None, "images": [decode_image(s["image"])]}
    raise ValueError(f"unknown experiment {exp}")


# Output row + save
def make_row(*, trial, exp, model, model_version, raw, parsed, ok,
             latency_ms, timestamp, run_id):
    return {
        "trial_id": trial["trial_id"],
        "experiment": exp,
        "model": model,
        "model_version": model_version,
        "stimulus_id": trial["stimulus_id"],
        "sample_idx": int(trial["sample_idx"]),
        "raw_response": raw,
        "parsed_value": parsed,
        "parse_ok": bool(ok),
        "scale": trial.get("scale"),
        "prompt_lang": trial.get("prompt_lang"),
        "option_order": trial.get("option_order"),
        "latency_ms": latency_ms,
        "timestamp": timestamp,
        "run_id": run_id,
    }


def save_rows(rows, out_path):
    import pandas as pd

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    tmp = out_path + ".tmp"
    if out_path.endswith(".csv"):
        df.to_csv(tmp, index=False)
    else:
        df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return len(df)


def load_rows(out_path):
    if not os.path.exists(out_path):
        return []
    import pandas as pd

    df = pd.read_csv(out_path) if out_path.endswith(".csv") else pd.read_parquet(out_path)
    return df.to_dict("records")


def shard_indices(n, shard_idx, shard_total):
    """Round-robin shard: trial i handled by rank (i % shard_total)."""
    return [i for i in range(n) if i % shard_total == shard_idx]


# Load API keys from .env (KEY=VALUE per line) into the environment.
def load_api_file(path=None):
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    paths = [path] if path else [os.path.join(root, ".env"), os.path.join(root, ".api")]
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                if key and key not in os.environ:
                    os.environ[key] = val.strip()
