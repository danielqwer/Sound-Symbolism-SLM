# Exp1 congruence with significance: per number, * = binomial vs chance 0.5 (choosing,
# not guessing); per model, bouba-vs-kiki congruence Wilcoxon across languages (dir + †).
import os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESP, OUT, CWIEK

LANGS = ["en", "ja", "zh-cn", "pt", "fr", "tr", "it", "de", "pl", "hu"]
MODELS = [("gemini-3.5-flash", "Gemini3.5-Flash"), ("gpt-audio-1.5", "GPT-Audio"),
          ("qwen3-omni", "Qwen3-Omni"), ("minicpm-o-4.5", "MiniCPM-o-4.5"),
          ("gemma-4-e4b", "Gemma4-E4B"), ("kimi-audio", "Kimi-Audio"),
          ("step-audio2", "Step-Audio2"), ("audio-flamingo3", "AudioFlamingo3")]
CW = {"en": "EN", "ja": "JP", "zh-cn": "CN", "pt": "PT", "fr": "FR",
      "tr": "TR", "it": "IT", "de": "DE", "pl": "PL", "hu": "HU"}


def num(k, n):
    return f"{k/n:.2f}" + ("*" if binomtest(k, n, 0.5).pvalue < .05 else "")


rows = []
for mk, mn in MODELS:
    d = pd.read_csv(f"{RESP}/{mk}/exp1.csv")
    d = d[(d.judge_ok == True) & (d.prompt_lang.isin(LANGS))]
    cells, xb, xk = [], [], []
    for l in LANGS:
        dl = d[d.prompt_lang == l]
        b = dl[dl.stimulus_id == "cwiek_bouba"]
        k = dl[dl.stimulus_id == "cwiek_kiki"]
        kb, nb = (b.judge_value == "rounded").sum(), len(b)
        kk, nk = (k.judge_value == "spiky").sum(), len(k)
        cells.append(f"{num(kb, nb)}/{num(kk, nk)}")
        xb.append(kb / nb)
        xk.append(kk / nk)
    xb, xk = np.array(xb), np.array(xk)
    p = wilcoxon(xb, xk).pvalue
    direc = ("b>k" if xb.mean() > xk.mean() else "k>b") + (" †" if p < .05 else "")
    rows.append([mn] + cells + [direc])

h = pd.read_csv(CWIEK)
hc = [f"{h[(h.Language==CW[l])&(h.Condition=='bouba')].ACC.mean():.2f}/"
      f"{h[(h.Language==CW[l])&(h.Condition=='kiki')].ACC.mean():.2f}" for l in LANGS]
rows.append(["Human"] + hc + ["-"])

out = pd.DataFrame(rows, columns=["Model"] + LANGS + ["bouba_vs_kiki"])
os.makedirs(OUT, exist_ok=True)
out.to_csv(f"{OUT}/exp1_congruence_significance.csv", index=False)
print(out.to_string(index=False))
print("\n* = binomial vs 0.5 (p<.05); † = bouba vs kiki congruence Wilcoxon across languages (p<.05)")
