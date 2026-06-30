import argparse
import datetime as dt
import importlib
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))   # scripts/
sys.path.insert(0, _HERE)                            # bk_common, config
sys.path.insert(0, os.path.dirname(_HERE))           # repo root: src/ package

import bk_common as bk          # noqa: E402
import config                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", required=True, choices=config.ALL_EXPS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--shard-total", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap #trials in this shard (for sanity runs)")
    ap.add_argument("--lang", default=None,
                    help="restrict to one prompt_lang (e.g. 'en') for targeted re-runs")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    bk.load_api_file()              # pull API keys from repo-root .api into env
    cfg = config.get(args.model)
    if args.exp not in cfg["exps"]:
        sys.exit(f"[skip] model '{args.model}' cannot run {args.exp} "
                 f"(modalities={sorted(cfg['modalities'])})")

    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    samp = config.SAMPLING
    max_new = config.EXP_MAX_NEW_TOKENS.get(args.exp, samp["max_new_tokens"])

    adapter = importlib.import_module(f"src.{cfg['module']}")
    print(f"[load] {args.model} via src.{cfg['module']} ...", flush=True)
    ctx = adapter.load(cfg)
    version = adapter.model_version(cfg, ctx)
    target_sr = getattr(adapter, "TARGET_SR", None)
    print(f"[load] done. model_version={version} target_sr={target_sr}",
          flush=True)

    trials = bk.load_trials(args.exp)
    if args.lang:                                   # targeted re-run subset
        keep = set(args.lang.split(","))
        trials = [t for t in trials if t.get("prompt_lang") in keep]
        print(f"[data] --lang {args.lang}: filtered to {len(trials)} trials", flush=True)
    stim_by_id = bk.load_stimuli(args.exp)
    idx = bk.shard_indices(len(trials), args.shard_idx, args.shard_total)
    if args.limit:
        idx = idx[: args.limit]
    print(f"[data] {args.exp}: {len(trials)} trials total, this shard "
          f"{args.shard_idx}/{args.shard_total} -> {len(idx)} trials", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rows = bk.load_rows(args.out)
    done = {r["trial_id"] for r in rows}
    idx = [i for i in idx if trials[i]["trial_id"] not in done]
    print(f"[resume] {len(done)} done, {len(idx)} remaining", flush=True)

    SAVE_EVERY = 25
    for n, i in enumerate(idx):
        t = trials[i]
        media = bk.build_media(args.exp, t, stim_by_id, target_sr)
        seed = int(t["trial_seed"])
        t0 = time.time()
        try:
            raw = adapter.generate(
                ctx,
                prompt=t["prompt_text"],
                audio=media["audio"],
                images=media["images"],
                temperature=samp["temperature"],
                seed=seed,
                max_new_tokens=max_new,
            )
        except Exception:
            raw = "ERROR:\n" + traceback.format_exc()
        latency_ms = int((time.time() - t0) * 1000)
        rows.append(bk.make_row(
            trial=t, exp=args.exp, model=args.model, model_version=version,
            raw=raw, parsed=None, ok=False, latency_ms=latency_ms,
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            run_id=run_id,
        ))
        if (n + 1) % SAVE_EVERY == 0:
            bk.save_rows(rows, args.out)
            print(f"  [{n+1}/{len(idx)}] saved", flush=True)

    written = bk.save_rows(rows, args.out)
    print(f"[done] wrote {written} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
