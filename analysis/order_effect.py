# Position/order effect (exp1 & exp3): P(chose round | round listed 1st) vs 2nd, Wilcoxon
# across languages. p_round = overall rate of the round response (response fixation).
import os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESP, OUT

L1 = ["en", "ja", "zh-cn", "pt", "fr", "tr", "it", "de", "pl", "hu"]
L3 = ["de", "en", "hy", "ro", "sq", "zh-cn", "pl", "fr", "th", "ja"]
E1 = [("gemini-3.5-flash", "Gemini3.5-Flash"), ("gpt-audio-1.5", "GPT-Audio"),
      ("qwen3-omni", "Qwen3-Omni"), ("minicpm-o-4.5", "MiniCPM-o-4.5"),
      ("gemma-4-e4b", "Gemma4-E4B"), ("kimi-audio", "Kimi-Audio"),
      ("step-audio2", "Step-Audio2"), ("audio-flamingo3", "AudioFlamingo3")]
E3 = [("gemini-3.5-flash", "Gemini3.5-Flash"), ("qwen3-omni", "Qwen3-Omni"),
      ("minicpm-o-4.5", "MiniCPM-o-4.5"), ("gemma-4-e4b", "Gemma4-E4B")]


def run(exp, models, langs, chose_round):
    rows = []
    for mk, mn in models:
        d = pd.read_csv(f"{RESP}/{mk}/{exp}.csv")
        d = d[(d.judge_ok == True) & (d.prompt_lang.isin(langs))].copy()
        d["cr"] = chose_round(d)
        a, b = [], []
        for lg in langs:
            dl = d[d.prompt_lang == lg]
            rf = dl[dl.option_order == "rounded_first"]
            sf = dl[dl.option_order == "spiky_first"]
            if len(rf) and len(sf):
                a.append(rf.cr.mean())
                b.append(sf.cr.mean())
        a, b = np.array(a), np.array(b)
        p = wilcoxon(a, b).pvalue
        rows.append([mn, round(d.cr.mean(), 3), round(a.mean(), 3), round(b.mean(), 3),
                     f"{p:.2g}", "†" if p < .05 else ""])
    return pd.DataFrame(rows, columns=["Model", "p_round", "round_1st", "round_2nd", "wilcoxon_p", "sig"])


t1 = run("exp1", E1, L1, lambda d: d.judge_value == "rounded")
t3 = run("exp3", E3, L3, lambda d: np.where(d.option_order == "rounded_first",
                                            d.judge_value == "A", d.judge_value == "B"))
os.makedirs(OUT, exist_ok=True)
t1.to_csv(f"{OUT}/exp1_order_effect.csv", index=False)
t3.to_csv(f"{OUT}/exp3_order_effect.csv", index=False)
print("EXP1 order effect (audio->text):")
print(t1.to_string(index=False))
print("\nEXP3 order effect (audio+image):")
print(t3.to_string(index=False))
