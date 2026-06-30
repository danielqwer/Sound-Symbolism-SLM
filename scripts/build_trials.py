import os
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bk_common as bk          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, os.pardir, "data", "grids")
OUT = os.path.join(HERE, os.pardir, "data", "trials")

BASE_COLS = ["trial_id", "prompt_lang", "scale", "stimulus_id",
             "sample_idx", "option_order", "prompt_text", "trial_seed"]

DROP_LANGS = {"zh-tw"}
EN_ONLY = {"en"}


def _grid(exp):
    return pd.DataFrame(pq.read_table(os.path.join(GRID, f"{exp}_trials.parquet")).to_pylist())


def build_exp1():
    g = _grid("exp1")
    g["scale"] = None
    return g[BASE_COLS + ["word_rounded", "word_spiky"]].copy()


def build_exp2():
    g = _grid("exp2")
    g = g[g.prompt_lang.isin(EN_ONLY)].copy()
    g["option_order"] = None
    stim = bk.load_stimuli("exp2")

    def ipa(sid):
        s = stim[sid]
        return "".join(str(s.get(k) or "") for k in ("ipa_C1", "ipa_V1", "ipa_C2", "ipa_V2"))

    g["phoneme"] = g.stimulus_id.map(ipa)
    return g[BASE_COLS + ["phoneme"]].copy()


def build_exp3():
    g = _grid("exp3")
    g = g[~g.prompt_lang.isin(DROP_LANGS)].copy()
    # fold the grid's 60/query (30 sidx x 2 positions) to 30/query, 15/15
    rows = []
    for (lang, sid), sub in g.groupby(["prompt_lang", "stimulus_id"]):
        for sidx in range(30):
            pos = "rounded_left" if sidx < 15 else "spiky_left"
            r = sub[(sub.sample_idx == sidx) & (sub.position_order == pos)]
            if len(r) != 1:
                raise RuntimeError(f"exp3 {lang}/{sid} sidx={sidx} pos={pos}: got {len(r)} rows")
            rows.append(r.iloc[0])
    out = pd.DataFrame(rows)
    out["scale"] = None
    out["option_order"] = out.position_order.map(
        {"rounded_left": "rounded_first", "spiky_left": "spiky_first"})
    return out[BASE_COLS].copy()


def build_exp4():
    g = _grid("exp4")
    g = g[g.prompt_lang.isin(EN_ONLY)].copy()
    g["option_order"] = None
    return g[BASE_COLS].copy()


def main():
    os.makedirs(OUT, exist_ok=True)
    builders = {"exp1": build_exp1, "exp2": build_exp2,
                "exp3": build_exp3, "exp4": build_exp4}
    for exp, fn in builders.items():
        df = fn()
        path = os.path.join(OUT, f"{exp}.csv")
        df.to_csv(path, index=False)
        oo = df.option_order.value_counts(dropna=False).to_dict() if "option_order" in df else {}
        print(f"[{exp}] {len(df):6d} rows  langs={df.prompt_lang.nunique()} "
              f"stim={df.stimulus_id.nunique()} option_order={oo} -> {path}")


if __name__ == "__main__":
    main()
