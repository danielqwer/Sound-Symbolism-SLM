import argparse
import glob
from pathlib import Path

import pandas as pd
import bk_common as bk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-glob", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = sorted(glob.glob(args.shards_glob))
    if not paths:
        ap.error("No shard files matched")
    if Path(args.out).resolve() in [Path(p).resolve() for p in paths]:
        ap.error("The output file must not also be an input shard")
    tables = []
    for path in paths:
        df = pd.read_parquet(path)
        missing = set(bk.OUTPUT_COLUMNS) - set(df.columns)
        if missing:
            ap.error(f"{path}: missing columns {sorted(missing)}")
        tables.append(df)
    out = pd.concat(tables, ignore_index=True)
    if out.empty:
        ap.error("All shards are empty")
    if out["trial_id"].isna().any() or out["trial_id"].duplicated().any():
        ap.error("Missing or duplicate trial IDs across shards")
    for field in ("experiment", "model", "model_version"):
        if out[field].nunique(dropna=False) != 1:
            ap.error(f"Shards disagree on {field}")
    Path(args.out).resolve().parent.mkdir(parents=True, exist_ok=True)
    bk.save_rows(out.sort_values("trial_id").to_dict("records"), args.out)
    print(f"Merged {len(paths)} shards, {len(out)} responses -> {args.out}")


if __name__ == "__main__":
    main()
