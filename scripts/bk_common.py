import io
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("BK_DATA_DIR") or ROOT.parent / "sound-symbolism-data").expanduser().resolve()


OUTPUT_COLUMNS = [
    "trial_id", "experiment", "model", "model_version", "stimulus_id",
    "sample_idx", "raw_response", "parsed_value", "parse_ok",

    "scale", "prompt_lang", "option_order", "latency_ms", "timestamp", "run_id",
]


def _local_trials_path(exp):
    return DATA / "trials" / f"{exp}.csv"


def load_trials(exp):
    import pandas as pd

    df = pd.read_csv(_local_trials_path(exp))
    df = df.astype(object).where(pd.notna(df), None)
    return df.to_dict("records")


def load_stimuli(exp):

    required = {"exp1": ("audio",), "exp2": ("audio",),
                "exp3": ("audio", "rounded_shape", "spiky_shape"),
                "exp4": ("image",)}
    if exp not in required:
        raise ValueError(f"unknown experiment {exp}")
    path = DATA / "stimuli" / f"{exp}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Prepare local stimuli with scripts/prepare_data.py; "
            "see README.md for data preparation.")
    stimuli = {}
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("stimulus_id")
            if not isinstance(sid, str) or not sid or sid in stimuli:
                raise ValueError(f"{path}:{line_no}: missing or duplicate stimulus_id")
            for field in required[exp]:
                media = row.get(field)
                if not isinstance(media, dict) or not media.get("path"):
                    raise ValueError(f"{path}:{line_no}: missing {field}.path")
                local = Path(media["path"])
                if not local.is_absolute():
                    local = path.parent / local
                local = local.resolve()
                if not local.is_file():
                    raise FileNotFoundError(f"{sid}: missing {field}: {local}")
                row[field] = {**media, "path": str(local)}
            stimuli[sid] = row
    if not stimuli:
        raise ValueError(f"Empty stimulus manifest: {path}")
    return stimuli


def media_source(media):

    return io.BytesIO(media["bytes"]) if "bytes" in media else media["path"]


def decode_audio(media, target_sr=None):

    import soundfile as sf

    arr, sr = sf.read(media_source(media), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = np.asarray(arr, dtype=np.float32)
    if target_sr and sr != target_sr:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return arr, sr


def decode_image(media):

    from PIL import Image

    with Image.open(media_source(media)) as im:
        return im.convert("RGB")


def build_media(exp, trial, stim_by_id, target_sr=None):
    sid = trial["stimulus_id"]
    s = stim_by_id[sid]
    if exp in ("exp1", "exp2"):
        return {"audio": decode_audio(s["audio"], target_sr), "images": None}
    if exp == "exp3":
        audio = decode_audio(s["audio"], target_sr)
        rounded = decode_image(s["rounded_shape"])
        spiky = decode_image(s["spiky_shape"])

        if trial["option_order"] == "rounded_first":
            images = [rounded, spiky]
        elif trial["option_order"] == "spiky_first":
            images = [spiky, rounded]
        else:
            raise ValueError(f"bad option_order: {trial['option_order']!r}")
        return {"audio": audio, "images": images}
    if exp == "exp4":
        return {"audio": None, "images": [decode_image(s["image"])]}
    raise ValueError(f"unknown experiment {exp}")


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

    return [i for i in range(n) if i % shard_total == shard_idx]


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
