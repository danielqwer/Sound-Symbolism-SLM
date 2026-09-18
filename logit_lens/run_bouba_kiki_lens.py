"""
Bouba-Kiki logit lens experiment — unified runner for MiniCPM and Qwen3-Omni.

exp2  — 537 audio stimuli (exp2_stimuli)
        × prompts INSTR-001 (rounded?) and INSTR-002 (pointed?)

exp5  — 90 image stimuli (abstract shapes)
        × prompts INSTR-017 (rounded?) and INSTR-018 (pointed?)

Usage:
    python run_bouba_kiki_lens.py --backend minicpm   --exp exp2
    python run_bouba_kiki_lens.py --backend qwen3-omni --exp exp5
    python run_bouba_kiki_lens.py --backend minicpm   --exp both --lang en
    python run_bouba_kiki_lens.py --backend qwen3-omni --exp exp2 --sample 20
    python run_bouba_kiki_lens.py --backend minicpm   --model /path/to/model --exp both
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer

from logit_lens import LogitLensAnalyzer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET = Path("/work/wilzwork23/bouba-kiki-lalm-unpacked")
PROMPTS = Path("/home/wilzwork23/boba_kiki_logit_lens/prompts_final.jsonl")
OUTPUT  = Path("/home/wilzwork23/boba_kiki_logit_lens/results")

_BACKEND_MODEL = {
    "minicpm":    "/work/wilzwork23/models/MiniCPM-o-4_5",
    "qwen3-omni": "/work/wilzwork23/models/Qwen3-Omni-30B-A3B-Instruct",
}

# ---------------------------------------------------------------------------
# Model loading — dispatches on --backend
# ---------------------------------------------------------------------------

def load_model(args):
    model_path = args.model or _BACKEND_MODEL[args.backend]
    dtype  = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    print(f"Loading [{args.backend}]: {model_path}  (dtype={args.dtype}, device={device})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if args.backend == "minicpm":
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(cfg, "tts_config") and cfg.tts_config is not None:
            for _k, _v in [
                ("top_p", 0.9), ("top_k", 50), ("repetition_penalty", 1.0),
                ("interleaved", False), ("attention_type", "full_attention"),
                ("recomputed_chunks", 1),
            ]:
                if not hasattr(cfg.tts_config, _k):
                    setattr(cfg.tts_config, _k, _v)
        model = AutoModel.from_pretrained(
            model_path, config=cfg, trust_remote_code=True,
            torch_dtype=dtype, device_map=device,
        )
    elif args.backend == "qwen3-omni":
        from transformers import Qwen3OmniMoeForConditionalGeneration
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend!r}")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Load prompts
# ---------------------------------------------------------------------------

def load_prompts(path: Path) -> dict:
    prompts = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            prompts[obj["id"]] = obj["prompt"]
    return prompts


def get_prompt(prompts: dict, instr_id: str, lang: str) -> str:
    key = f"{instr_id}_{lang}"
    if key not in prompts:
        raise KeyError(f"Prompt '{key}' not found. Available: {list(prompts.keys())[:6]} …")
    return prompts[key]


# ---------------------------------------------------------------------------
# Path resolver
# ---------------------------------------------------------------------------

def resolve_path(root: Path, stimulus_id: str, csv_path: str, ext: str) -> Path:
    csv_stem = Path(csv_path).stem
    csv_name = Path(csv_path).name
    candidates = [
        root / f"{stimulus_id}{ext}",
        root / f"{csv_stem}{ext}",
        root / csv_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Cannot find file for stimulus '{stimulus_id}'.\nTried:\n  {tried}"
    )


# ---------------------------------------------------------------------------
# Load audio
# ---------------------------------------------------------------------------

def load_wav(path: Path, target_sr: int = 16_000) -> np.ndarray:
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
    except ImportError:
        import librosa
        data, sr = librosa.load(str(path), sr=target_sr, mono=True)
        return data
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_digit_heatmap(layer_probs, title, save_path, digits="1234567"):
    layers   = sorted(layer_probs.keys())
    n_layers = len(layers)
    n_digits = len(digits)

    mat = np.zeros((n_digits, n_layers))
    for li, lk in enumerate(layers):
        for di, d in enumerate(digits):
            mat[di, li] = layer_probs[lk].get(d, 0.0)

    fig, ax = plt.subplots(figsize=(max(12, n_layers * 0.35), 4))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0.0)
    ax.set_yticks(range(n_digits))
    ax.set_yticklabels([f'"{d}"' for d in digits], fontsize=10)
    ax.set_xticks(range(0, n_layers, max(1, n_layers // 20)))
    ax.set_xticklabels(
        [f"L{layers[i]}" for i in range(0, n_layers, max(1, n_layers // 20))],
        fontsize=8, rotation=45,
    )
    ax.set_xlabel("Layer  (L0 = embedding)", fontsize=11)
    ax.set_ylabel("Digit token", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, label="Softmax probability", shrink=0.8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_digit_lines(layer_probs_dict, title, save_path, digits="1234567"):
    any_lp = next(iter(layer_probs_dict.values()))
    layers = sorted(any_lp.keys())

    fig, axes = plt.subplots(1, len(digits), figsize=(len(digits) * 3, 3.5), sharey=True)
    colors = plt.cm.tab10.colors

    for di, d in enumerate(digits):
        ax = axes[di]
        for ci, (label, lp) in enumerate(layer_probs_dict.items()):
            ys = [lp[lk].get(d, 0.0) for lk in layers]
            ax.plot(layers, ys, label=label, color=colors[ci % len(colors)], linewidth=1.5)
        ax.set_title(f'P("{d}")', fontsize=10)
        ax.set_xlabel("Layer", fontsize=8)
        if di == 0:
            ax.set_ylabel("Probability", fontsize=9)
        ax.set_xticks(layers[::max(1, len(layers) // 5)])
        ax.tick_params(labelsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.7)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_aggregate_heatmap(group_layer_probs, title, save_path, digits="1234567"):
    groups = list(group_layer_probs.keys())
    fig, axes = plt.subplots(1, len(groups),
                             figsize=(len(groups) * 8, 4), sharey=True)
    if len(groups) == 1:
        axes = [axes]

    for ax, grp in zip(axes, groups):
        lp     = group_layer_probs[grp]
        layers = sorted(lp.keys())
        mat    = np.zeros((len(digits), len(layers)))
        for li, lk in enumerate(layers):
            for di, d in enumerate(digits):
                mat[di, li] = lp[lk].get(d, 0.0)
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0.0)
        ax.set_title(grp, fontsize=11)
        ax.set_yticks(range(len(digits)))
        ax.set_yticklabels([f'"{d}"' for d in digits], fontsize=9)
        ax.set_xticks(range(0, len(layers), max(1, len(layers) // 10)))
        ax.set_xticklabels(
            [f"L{layers[i]}" for i in range(0, len(layers), max(1, len(layers) // 10))],
            fontsize=7, rotation=45,
        )
        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# GT group builder
# ---------------------------------------------------------------------------

def build_gt_groups_from_stimuli(meta_csv, rating_col, thresholds=(3.0, 5.0)):
    """
    GT groups from per-stimulus human ratings (roundedness_mean / pointedness_mean,
    1–7 scale) in the stimuli CSV — one value per stimulus, no trials averaging needed.
    """
    df  = pd.read_csv(meta_csv)
    avg = df.set_index("stimulus_id")[rating_col].dropna()

    t_lo, t_hi = thresholds
    low_ids  = set(avg[avg <  t_lo].index)
    mid_ids  = set(avg[(avg >= t_lo) & (avg < t_hi)].index)
    high_ids = set(avg[avg >= t_hi].index)

    human_stats = {}
    for band, ids in [("1-3", low_ids), ("3-5", mid_ids), ("5-7", high_ids)]:
        if ids:
            vals = avg[avg.index.isin(ids)]
            human_stats[band] = (float(vals.mean()), float(vals.std()), len(ids))
        else:
            human_stats[band] = (float("nan"), float("nan"), 0)

    print(
        f"    GT [{rating_col}]:  <{t_lo} (1-3): {len(low_ids)}  |  "
        f"{t_lo}–{t_hi} (3-5): {len(mid_ids)}  |  ≥{t_hi} (5-7): {len(high_ids)}"
    )
    return low_ids, mid_ids, high_ids, human_stats


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def avg_layer_probs(lp_list):
    if not lp_list:
        return {}
    result = {}
    for lk in lp_list[0]:
        result[lk] = {}
        for d in lp_list[0][lk]:
            result[lk][d] = np.mean([lp[lk][d] for lp in lp_list if lk in lp])
    return result


def avg_argmax_with_var(lp_list, digits="1234567"):
    if not lp_list:
        return {}
    all_layers = sorted(set().union(*[set(lp.keys()) for lp in lp_list]))
    result = {}
    for lk in all_layers:
        values = []
        for lp in lp_list:
            if lk not in lp:
                continue
            best = max(digits, key=lambda d: lp[lk].get(d, 0.0))
            values.append(int(best))
        if values:
            result[lk] = (float(np.mean(values)), float(np.std(values)))
    return result


def plot_argmax_line(group_data, title, save_path, human_stats=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = plt.cm.tab10.colors
    for ci, (label, layer_vals) in enumerate(group_data.items()):
        if not layer_vals:
            continue
        layers     = sorted(layer_vals.keys())
        sample_val = layer_vals[layers[0]]
        has_std    = isinstance(sample_val, (tuple, list))

        if has_std:
            ys     = np.array([layer_vals[lk][0] for lk in layers])
            ys_std = np.array([layer_vals[lk][1] for lk in layers])
        else:
            ys     = np.array([layer_vals[lk] for lk in layers])
            ys_std = None

        color        = colors[ci % len(colors)]
        legend_label = label
        if human_stats and label in human_stats:
            hm, hs, hn = human_stats[label]
            if not (isinstance(hm, float) and np.isnan(hm)):
                legend_label = f"{label}  [human μ={hm:.2f}±{hs:.2f}]"

        ax.plot(layers, ys, label=legend_label,
                color=color, linewidth=2, marker="o", markersize=3)
        if ys_std is not None:
            ax.fill_between(
                layers,
                np.clip(ys - ys_std, 1, 7),
                np.clip(ys + ys_std, 1, 7),
                alpha=0.2, color=color,
            )

    ax.set_xlabel("Layer  (0 = embedding)", fontsize=11)
    ax.set_ylabel("Mean argmax digit  (1–7)", fontsize=11)
    ax.set_yticks(range(1, 8))
    ax.set_ylim(0.5, 7.5)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ---------------------------------------------------------------------------
# exp2: audio
# ---------------------------------------------------------------------------

def run_exp2(analyzer, prompts, lang, out, backend,
             sample=0, num_workers=1, worker_id=0):
    print("\n====== EXP2 — Audio (537 phonetic stimuli) ======")

    audio_dir  = DATASET / "audio" / "exp2"
    meta_csv   = DATASET / "csv" / "exp2_stimuli.csv"
    trials_csv = DATASET / "csv" / "exp2_trials.csv"
    meta       = pd.read_csv(meta_csv)

    if sample > 0:
        meta = meta.head(sample)
        print(f"  Subsampling to first {sample} audio files.")
    if num_workers > 1:
        meta = meta.iloc[worker_id::num_workers].reset_index(drop=True)
        print(f"  Worker {worker_id}/{num_workers}: {len(meta)} stimuli assigned.")

    print("  Loading GT groups from exp2_stimuli.csv (roundedness_mean / pointedness_mean) …")
    round_low, round_mid, round_high, round_hstats = build_gt_groups_from_stimuli(
        meta_csv, "roundedness_mean"
    )
    point_low, point_mid, point_high, point_hstats = build_gt_groups_from_stimuli(
        meta_csv, "pointedness_mean"
    )

    instr_gt = {
        "INSTR-001": (round_low, round_mid, round_high, round_hstats, "roundedness"),
        "INSTR-002": (point_low, point_mid, point_high, point_hstats, "pointedness"),
    }
    collected = {k: {"low": [], "mid": [], "high": []} for k in instr_gt}

    for _, row in meta.iterrows():
        sid = row["stimulus_id"]
        try:
            wav_path = resolve_path(audio_dir, sid, row["audio_path"], ".wav")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}\n  Skipping.")
            continue

        audio = load_wav(wav_path)
        print(f"  Audio: {wav_path.name}")

        for instr_id, (low_ids, mid_ids, high_ids, h_stats, dim) in instr_gt.items():
            prompt_text = get_prompt(prompts, instr_id, lang)
            lp = analyzer.probe_digit_probs(text=prompt_text, audio=[audio])

            if sid in high_ids:
                collected[instr_id]["high"].append(lp)
                gt_label = "5-7"
            elif sid in mid_ids:
                collected[instr_id]["mid"].append(lp)
                gt_label = "3-5"
            elif sid in low_ids:
                collected[instr_id]["low"].append(lp)
                gt_label = "1-3"
            else:
                gt_label = "no-gt"

            if sample > 0:
                plot_digit_heatmap(
                    lp,
                    title=(f"[exp2][{backend}] {sid} | {instr_id} ({dim}) | "
                           f"human={gt_label} | lang={lang}"),
                    save_path=out / "exp2" / "per_audio" / f"{sid}__{instr_id}.png",
                )

    raw_dir  = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"worker_{worker_id}_exp2.jsonl"
    with open(raw_path, "w") as _f:
        for _instr_id, _buckets in collected.items():
            for _gt_grp, _lp_list in _buckets.items():
                for _lp in _lp_list:
                    _f.write(json.dumps({
                        "exp": "exp2", "instr_id": _instr_id, "gt_group": _gt_grp,
                        "backend": backend,
                        "layer_probs": {str(lk): v for lk, v in _lp.items()},
                    }) + "\n")
    print(f"  Saved raw → {raw_path}")

    if num_workers == 1:
        _BANDS = [("1-3", "low"), ("3-5", "mid"), ("5-7", "high")]
        for instr_id, (_, _, _, h_stats, dim) in instr_gt.items():
            group_line, group_heat, hstats_plot = {}, {}, {}
            for band, bucket in _BANDS:
                items = collected[instr_id][bucket]
                n = len(items)
                if n:
                    key = f"{band}  (n={n})"
                    group_line[key]  = avg_argmax_with_var(items)
                    group_heat[key]  = avg_layer_probs(items)
                    hstats_plot[key] = h_stats[band]
            base = f"[exp2][{backend}] {instr_id} — {dim} | lang={lang}"
            if group_line:
                plot_argmax_line(
                    group_line,
                    title=f"{base}\nMean argmax digit by human score band (±1 std shaded)",
                    save_path=out / "exp2" / f"gt_split__{instr_id}_line.png",
                    human_stats=hstats_plot,
                )
            if group_heat:
                plot_aggregate_heatmap(
                    group_heat,
                    title=f"{base}\nAvg softmax by human score band",
                    save_path=out / "exp2" / f"gt_split__{instr_id}_heat.png",
                )

    print("\n  EXP2 done.")


# ---------------------------------------------------------------------------
# exp5: images
# ---------------------------------------------------------------------------

def run_exp5(analyzer, prompts, lang, out, backend,
             sample=0, num_workers=1, worker_id=0):
    print("\n====== EXP5 — Images (abstract shapes) ======")

    img_dir    = DATASET / "images" / "exp5_png"
    meta_csv   = DATASET / "csv" / "exp5_stimuli.csv"
    trials_csv = DATASET / "csv" / "exp5_trials.csv"
    meta       = pd.read_csv(meta_csv)

    if sample > 0:
        meta = meta.head(sample)
        print(f"  Subsampling to first {sample} images.")
    if num_workers > 1:
        meta = meta.iloc[worker_id::num_workers].reset_index(drop=True)
        print(f"  Worker {worker_id}/{num_workers}: {len(meta)} stimuli assigned.")

    print("  Loading GT groups from exp5_stimuli.csv (roundedness_mean / pointedness_mean) …")
    round_low, round_mid, round_high, round_hstats = build_gt_groups_from_stimuli(
        meta_csv, "roundedness_mean"
    )
    point_low, point_mid, point_high, point_hstats = build_gt_groups_from_stimuli(
        meta_csv, "pointedness_mean"
    )

    instr_gt = {
        "INSTR-017": (round_low, round_mid, round_high, round_hstats, "rounded"),
        "INSTR-018": (point_low, point_mid, point_high, point_hstats, "pointed"),
    }
    collected = {k: {"low": [], "mid": [], "high": []} for k in instr_gt}

    for _, row in meta.iterrows():
        sid = row["stimulus_id"]
        try:
            img_file = resolve_path(img_dir, sid, row["image_path"], ".png")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}\n  Skipping.")
            continue

        pil_img = Image.open(img_file).convert("RGB")
        print(f"  Image: {img_file.name}")

        for instr_id, (low_ids, mid_ids, high_ids, h_stats, dim) in instr_gt.items():
            prompt_text = get_prompt(prompts, instr_id, lang)
            lp = analyzer.probe_digit_probs(text=prompt_text, images=[pil_img])

            if sid in high_ids:
                collected[instr_id]["high"].append(lp)
                gt_label = "5-7"
            elif sid in mid_ids:
                collected[instr_id]["mid"].append(lp)
                gt_label = "3-5"
            elif sid in low_ids:
                collected[instr_id]["low"].append(lp)
                gt_label = "1-3"
            else:
                gt_label = "no-gt"

            if sample > 0:
                plot_digit_heatmap(
                    lp,
                    title=(f"[exp5][{backend}] {sid} | {instr_id} ({dim}) | "
                           f"human={gt_label} | lang={lang}"),
                    save_path=out / "exp5" / "per_image" / f"{sid}__{instr_id}.png",
                )

    raw_dir  = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"worker_{worker_id}_exp5.jsonl"
    with open(raw_path, "w") as _f:
        for _instr_id, _buckets in collected.items():
            for _gt_grp, _lp_list in _buckets.items():
                for _lp in _lp_list:
                    _f.write(json.dumps({
                        "exp": "exp5", "instr_id": _instr_id, "gt_group": _gt_grp,
                        "backend": backend,
                        "layer_probs": {str(lk): v for lk, v in _lp.items()},
                    }) + "\n")
    print(f"  Saved raw → {raw_path}")

    if num_workers == 1:
        _BANDS = [("1-3", "low"), ("3-5", "mid"), ("5-7", "high")]
        for instr_id, (_, _, _, h_stats, dim) in instr_gt.items():
            group_line, group_heat, hstats_plot = {}, {}, {}
            for band, bucket in _BANDS:
                items = collected[instr_id][bucket]
                n = len(items)
                if n:
                    key = f"{band}  (n={n})"
                    group_line[key]  = avg_argmax_with_var(items)
                    group_heat[key]  = avg_layer_probs(items)
                    hstats_plot[key] = h_stats[band]
            base = f"[exp5][{backend}] {instr_id} — {dim} | lang={lang}"
            if group_line:
                plot_argmax_line(
                    group_line,
                    title=f"{base}\nMean argmax digit by human score band (±1 std shaded)",
                    save_path=out / "exp5" / f"gt_split__{instr_id}_line.png",
                    human_stats=hstats_plot,
                )
            if group_heat:
                plot_aggregate_heatmap(
                    group_heat,
                    title=f"{base}\nAvg softmax by human score band",
                    save_path=out / "exp5" / f"gt_split__{instr_id}_heat.png",
                )

    print("\n  EXP5 done.")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def run_merge(out, exp, lang, backend):
    raw_dir = out / "raw"
    exps = ["exp2", "exp5"] if exp == "both" else [exp]

    _instr_meta = {
        "exp2": {
            "INSTR-001": ("roundedness", DATASET / "csv" / "exp2_stimuli.csv", "roundedness_mean"),
            "INSTR-002": ("pointedness", DATASET / "csv" / "exp2_stimuli.csv", "pointedness_mean"),
        },
        "exp5": {
            "INSTR-017": ("rounded",     DATASET / "csv" / "exp5_stimuli.csv", "roundedness_mean"),
            "INSTR-018": ("pointed",     DATASET / "csv" / "exp5_stimuli.csv", "pointedness_mean"),
        },
    }
    _BANDS = [("1-3", "low"), ("3-5", "mid"), ("5-7", "high")]

    for e in exps:
        print(f"\n====== MERGE {e.upper()} [{backend}] ======")
        instr_meta = _instr_meta[e]
        collected  = {k: {"low": [], "mid": [], "high": []} for k in instr_meta}
        n_files    = 0

        for jsonl_f in sorted(raw_dir.glob(f"*_{e}.jsonl")):
            n_files += 1
            with open(jsonl_f) as f:
                for line in f:
                    rec = json.loads(line)
                    # Filter by backend if the record carries it
                    if "backend" in rec and rec["backend"] != backend:
                        continue
                    iid = rec["instr_id"]
                    grp = rec["gt_group"]
                    if iid not in collected or grp not in ("low", "mid", "high"):
                        continue
                    lp = {int(lk): v for lk, v in rec["layer_probs"].items()}
                    collected[iid][grp].append(lp)

        if n_files == 0:
            print(f"  No raw JSONL files found in {raw_dir} for {e}.")
            continue

        print(f"  Loaded {n_files} JSONL file(s).")

        for iid, (dim, meta_csv, rating_col) in instr_meta.items():
            _, _, _, h_stats = build_gt_groups_from_stimuli(meta_csv, rating_col)
            group_line, group_heat, hstats_plot = {}, {}, {}
            for band, bucket in _BANDS:
                items = collected[iid][bucket]
                n = len(items)
                if n:
                    key = f"{band}  (n={n})"
                    group_line[key]  = avg_argmax_with_var(items)
                    group_heat[key]  = avg_layer_probs(items)
                    hstats_plot[key] = h_stats[band]
            base = f"[{e}][{backend}] {iid} — {dim} | lang={lang}"
            if group_line:
                plot_argmax_line(
                    group_line,
                    title=f"{base}\nMean argmax digit by human score band (±1 std shaded)",
                    save_path=out / e / f"gt_split__{iid}_line.png",
                    human_stats=hstats_plot,
                )
            if group_heat:
                plot_aggregate_heatmap(
                    group_heat,
                    title=f"{base}\nAvg softmax by human score band",
                    save_path=out / e / f"gt_split__{iid}_heat.png",
                )

        print(f"  MERGE {e.upper()} done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Bouba-Kiki logit lens")
    p.add_argument("--backend", choices=["minicpm", "qwen3-omni"], default="minicpm",
                   help="Model backend: minicpm | qwen3-omni")
    p.add_argument("--model",  default=None,
                   help="Override model path (default: auto from --backend)")
    p.add_argument("--exp",    choices=["exp2", "exp5", "both"], default="both")
    p.add_argument("--lang",   default="en")
    p.add_argument("--sample", type=int, default=0,
                   help="Limit stimuli to first N (0 = all)")
    p.add_argument("--dtype",  choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=str(OUTPUT))
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--worker_id",   type=int, default=0)
    p.add_argument("--mode", choices=["worker", "merge"], default="worker")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "merge":
        run_merge(out, args.exp, args.lang, args.backend)
        return

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Device : {device} | dtype : {args.dtype} | "
          f"backend : {args.backend} | worker {args.worker_id}/{args.num_workers}")

    model, tokenizer = load_model(args)
    prompts  = load_prompts(PROMPTS)
    analyzer = LogitLensAnalyzer(model, tokenizer)

    if args.exp in ("exp2", "both"):
        run_exp2(analyzer, prompts, args.lang, out, args.backend,
                 sample=args.sample, num_workers=args.num_workers, worker_id=args.worker_id)

    if args.exp in ("exp5", "both"):
        run_exp5(analyzer, prompts, args.lang, out, args.backend,
                 sample=args.sample, num_workers=args.num_workers, worker_id=args.worker_id)

    print(f"\nWorker {args.worker_id} done. Raw results in {out}/raw/")


if __name__ == "__main__":
    main()
