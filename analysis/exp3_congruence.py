from pathlib import Path

import numpy as np
import pandas as pd

from paths import RESP, OUT, CWIEK

LANGS = ["de", "en", "hy", "ro", "sq", "zh-cn", "pl", "fr", "th", "ja"]
HUMAN_LANG = dict(zip(LANGS, ["DE", "EN", "AM", "RO", "AL", "CN", "PL", "FR", "TH", "JP"]))
MODELS = ["gemini-3.5-flash", "qwen3-omni", "minicpm-o-4.5", "gemma-4-e4b"]


def congruence(df):
    valid = df[(df.judge_ok == True) & df.prompt_lang.isin(LANGS)].copy()
    if not valid.option_order.isin(["rounded_first", "spiky_first"]).all():
        raise ValueError("Unknown option order in Experiment 3 responses")
    if not valid.judge_value.isin(["A", "B"]).all():
        raise ValueError("Expected A/B values for valid Experiment 3 responses")
    valid["rounded"] = np.where(valid.option_order == "rounded_first",
                                valid.judge_value == "A", valid.judge_value == "B")
    rows = []
    for lang in LANGS:
        d = valid[valid.prompt_lang == lang]
        b = d[d.stimulus_id == "cwiek_bouba"]
        k = d[d.stimulus_id == "cwiek_kiki"]
        rows.append(dict(language=lang, bouba_match=b.rounded.mean(),
                         kiki_match=(~k.rounded).mean(), n_bouba=len(b), n_kiki=len(k)))
    return rows


def main():
    rows = []
    for model in MODELS:
        rows.extend(dict(model=model, **r) for r in congruence(pd.read_csv(Path(RESP) / model / "exp3.csv")))
    human = pd.read_csv(CWIEK)
    for lang in LANGS:
        d = human[human.Language == HUMAN_LANG[lang]]
        b, k = d[d.Condition == "bouba"], d[d.Condition == "kiki"]
        rows.append(dict(model="Human", language=lang, bouba_match=b.ACC.mean(),
                         kiki_match=k.ACC.mean(), n_bouba=len(b), n_kiki=len(k)))
    out = pd.DataFrame(rows)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(OUT) / "exp3_congruence_per_language.csv", index=False)
    summary = out.groupby("model")[["bouba_match", "kiki_match"]].mean()
    summary.to_csv(Path(OUT) / "exp3_congruence_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
