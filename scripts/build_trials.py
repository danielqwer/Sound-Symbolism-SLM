import argparse
import csv
import json
from pathlib import Path

import bk_common as bk

ROOT = Path(__file__).resolve().parents[1]
BASE_COLS = ["trial_id", "prompt_lang", "scale", "stimulus_id", "sample_idx",
             "option_order", "prompt_text", "trial_seed"]


def build(exp, stimuli, prompts, conditions):
    rows = []
    if exp == "exp1":
        for si, sid in enumerate(("cwiek_bouba", "cwiek_kiki")):
            if sid not in stimuli:
                raise ValueError(f"Missing stimulus: {sid}")
            for c in conditions:
                for offset in range(15):
                    index = si * 780 + c["trial_index_start"] + offset
                    row = {k: c[k] for k in ("prompt_lang", "option_order", "prompt_text",
                                             "word_rounded", "word_spiky")}
                    rows.append(dict(row, trial_id=f"e1_{index:06d}", scale=None,
                                     stimulus_id=sid, sample_idx=c["sample_start"] + offset,
                                     trial_seed=42 + index))
    elif exp == "exp3":
        langs = sorted({c["prompt_lang"] for c in conditions})
        for lang in langs:
            for si, sid in enumerate(("cwiek_bouba", "cwiek_kiki")):
                if sid not in stimuli:
                    raise ValueError(f"Missing stimulus: {sid}")
                for sample in range(30):
                    index = si * 60 + sample + (30 if sample >= 15 else 0)
                    rows.append(dict(trial_id=f"e3_{index:05d}_{lang}", prompt_lang=lang,
                                     scale=None, stimulus_id=sid, sample_idx=sample,
                                     option_order="rounded_first" if sample < 15 else "spiky_first",
                                     prompt_text=prompts[f"INSTR-016_{lang}"], trial_seed=42 + index))
    else:
        if exp == "exp2":
            order = sorted(stimuli)
            scales = [("roundedness", "INSTR-001_en"), ("pointedness", "INSTR-002_en")]
        else:
            order = [f"mccormick_shape_{letter}{i}" for letter in "ABCDEF" for i in range(1, 16)]
            scales = [("rounded", "INSTR-017_en"), ("pointed", "INSTR-018_en")]
        for sid in order:
            if sid not in stimuli:
                raise ValueError(f"Missing stimulus: {sid}")
            for scale, prompt in scales:
                for sample in range(30):
                    index = len(rows)

                    tid = f"e2_{index:07d}_en" if exp == "exp2" else f"e5_{index:05d}_en"
                    row = dict(trial_id=tid, prompt_lang="en", scale=scale,
                               stimulus_id=sid, sample_idx=sample, option_order=None,
                               prompt_text=prompts[prompt], trial_seed=42 + index)
                    if exp == "exp2":
                        s = stimuli[sid]
                        row["phoneme"] = s.get("phoneme") or "".join(
                            str(s.get(k) or "") for k in ("ipa_C1", "ipa_V1", "ipa_C2", "ipa_V2"))
                    rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exps", nargs="+", choices=("exp1", "exp2", "exp3", "exp4"),
                        default=["exp1", "exp2", "exp3", "exp4"])
    args = parser.parse_args()
    settings = json.loads((ROOT / "prompts/prompts.json").read_text(encoding="utf-8"))
    prompts = {r["id"]: r["prompt"] for r in settings["templates"]}
    conditions = settings["exp1_conditions"]
    out = bk.DATA / "trials"
    out.mkdir(parents=True, exist_ok=True)
    for exp in args.exps:
        stimuli = bk.load_stimuli(exp)
        expected = {"exp1": 2, "exp2": 537, "exp3": 2, "exp4": 90}[exp]
        if len(stimuli) != expected:
            raise ValueError(f"{exp}: expected {expected} paper stimuli, got {len(stimuli)}")
        rows = build(exp, stimuli, prompts, conditions)
        fields = BASE_COLS + ({"exp1": ["word_rounded", "word_spiky"], "exp2": ["phoneme"]}.get(exp, []))
        with (out / f"{exp}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{exp}: {len(rows)} trials -> {out / (exp + '.csv')}")


if __name__ == "__main__":
    main()
