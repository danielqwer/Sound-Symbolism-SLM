import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESP, OUT, CWIEK

LANGS = ["en", "ja", "zh-cn", "pt", "fr", "tr", "it", "de", "pl", "hu"]
MODELS = [("gemini-3.5-flash", "Gemini3.5-Flash"), ("gpt-audio-1.5", "GPT-Audio"),
          ("qwen3-omni", "Qwen3-Omni"), ("minicpm-o-4.5", "MiniCPM-o-4.5"),
          ("gemma-4-e4b", "Gemma4-E4B"), ("audio-flamingo3", "AudioFlamingo3"),
          ("kimi-audio", "Kimi-Audio"), ("step-audio2", "Step-Audio2")]
CW = {"en": "EN", "ja": "JP", "zh-cn": "CN", "pt": "PT", "fr": "FR",
      "tr": "TR", "it": "IT", "de": "DE", "pl": "PL", "hu": "HU"}


def model_rs(mdl, lang):
    d = pd.read_csv(f"{RESP}/{mdl}/exp1.csv")
    d = d[d.prompt_lang == lang]
    b = d[(d.stimulus_id == "cwiek_bouba") & (d.judge_ok == True)]
    k = d[(d.stimulus_id == "cwiek_kiki") & (d.judge_ok == True)]
    return (b.judge_value == "rounded").mean(), (k.judge_value == "spiky").mean()


def human_rs(lang):
    h = pd.read_csv(CWIEK)
    h = h[h.Language == CW[lang]]
    return h[h.Condition == "bouba"].ACC.mean(), h[h.Condition == "kiki"].ACC.mean()


rs = {mn: {l: model_rs(mk, l) for l in LANGS} for mk, mn in MODELS}
rs["Human"] = {l: human_rs(l) for l in LANGS}
order = [mn for _, mn in MODELS] + ["Human"]

comb = pd.DataFrame({l: {n: f"{round(rs[n][l][0],2)}/{round(rs[n][l][1],2)}" for n in order} for l in LANGS}).loc[order, LANGS]
rnd = pd.DataFrame({l: {n: round(rs[n][l][0], 3) for n in order} for l in LANGS}).loc[order, LANGS]
spk = pd.DataFrame({l: {n: round(rs[n][l][1], 3) for n in order} for l in LANGS}).loc[order, LANGS]

os.makedirs(OUT, exist_ok=True)
comb.to_csv(f"{OUT}/exp1_congruence_combined_top10lang.csv")
rnd.to_csv(f"{OUT}/exp1_round_congruence_top10lang.csv")
spk.to_csv(f"{OUT}/exp1_spiky_congruence_top10lang.csv")
print(comb.to_string())
