# Exp2 first-order model/human combine correlation (Pearson + Spearman).
import os, sys, re, warnings
import numpy as np, pandas as pd, scipy.io as sio
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LACEY, RESP, OUT


def cell(x):
    while isinstance(x, np.ndarray) and x.size >= 1:
        x = x.ravel()[0]
    return str(x)


def base(w):
    return re.match(r"Start-(\d+)", str(w).replace("mccormick_", "").replace("_mono", "")).group(1)


fn = sio.loadmat(f"{LACEY}/pseudowords537_Final_YJ.mat")["filenames_Final"]
ff = [base(cell(fn[i, 0])) for i in range(537)]
A = sio.loadmat(f"{LACEY}/RSA_Ordered_P_to_R_culled.mat")["combo_Final_Order_culled"]  # 31 x 537
ROUND, POINT = np.arange(0, 17), np.arange(17, 31)
hcomb = (np.nanmean(A[ROUND], 0) - np.nanmean(A[POINT], 0) + 8) / 2.0

keep = [i for i, b in enumerate(ff) if b != "253"]              # 536 shared words
order = [ff[i] for i in keep]
pos = {b: k for k, b in enumerate(order)}
n = len(order)
hvec = hcomb[keep]


def model_combine(mdl):
    df = pd.read_csv(f"{RESP}/{mdl}/exp2.csv")
    df = df[df.judge_ok == True].copy()
    df["base"] = df.stimulus_id.map(base)
    df["jv"] = pd.to_numeric(df.judge_value, errors="coerce")
    g = df.groupby(["base", "scale"]).jv.mean().unstack("scale")
    comb = (g["roundedness"] - g["pointedness"] + 8) / 2
    v = np.full(n, np.nan)
    for b, val in comb.items():
        if b in pos:
            v[pos[b]] = val
    return v


def sig(p):
    return "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "n.s."))


def ceiling(corrfn, nperm=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nperm):
        r, p = rng.permutation(ROUND), rng.permutation(POINT)
        c1 = (np.nanmean(A[r[:len(r)//2]], 0) - np.nanmean(A[p[:len(p)//2]], 0) + 8) / 2
        c2 = (np.nanmean(A[r[len(r)//2:]], 0) - np.nanmean(A[p[len(p)//2:]], 0) + 8) / 2
        c1, c2 = c1[keep], c2[keep]
        ok = ~(np.isnan(c1) | np.isnan(c2))
        vals.append(corrfn(c1[ok], c2[ok])[0])
    rh = np.mean(vals)
    return 2 * rh / (1 + rh)


MODELS = [("qwen3-omni", "Qwen3-Omni"), ("kimi-audio", "Kimi-Audio"), ("gemini-3.5-flash", "Gemini3.5-Flash"),
          ("minicpm-o-4.5", "MiniCPM-o-4.5"), ("gpt-audio-1.5", "GPT-Audio"), ("gemma-4-e4b", "Gemma4-E4B"),
          ("step-audio2", "Step-Audio2"), ("audio-flamingo3", "AudioFlamingo3")]

rows = [["Human ceiling", f"{ceiling(pearsonr):.3f}", "-", "-", f"{ceiling(spearmanr):.3f}", "-", "-", "-"]]
res = []
for mk, mn in MODELS:
    m = model_combine(mk)
    msk = ~(np.isnan(hvec) | np.isnan(m))
    pr, pp = pearsonr(hvec[msk], m[msk])
    sr, sp = spearmanr(hvec[msk], m[msk])
    res.append((mn, pr, pp, sr, sp, int(msk.sum())))
res.sort(key=lambda x: -x[1])
for mn, pr, pp, sr, sp, N in res:
    rows.append([mn, f"{pr:+.3f}", f"{pp:.2g}", sig(pp), f"{sr:+.3f}", f"{sp:.2g}", sig(sp), N])

out = pd.DataFrame(rows, columns=["Model", "pearson_r", "pearson_p", "pearson_sig",
                                  "spearman_rho", "spearman_p", "spearman_sig", "N"])
os.makedirs(OUT, exist_ok=True)
out.to_csv(f"{OUT}/exp2_combined_human_correlation.csv", index=False)
print(out.to_string(index=False))
