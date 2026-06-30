# Exp4 first-order model/human shape-combine correlation (Pearson + Spearman).
import os, sys
import pandas as pd, numpy as np, scipy.io as sio
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "scripts"))
import bk_common as bk
from paths import LACEY, RESP, OUT

st = bk.load_stimuli("exp4")
hcomb = {s: (st[s]["roundedness_mean"] - st[s]["pointedness_mean"] + 8) / 2 for s in st}


def model_combine(mdl):
    d = pd.read_csv(f"{RESP}/{mdl}/exp4.csv")
    d = d[d.judge_ok == True].copy()
    d["v"] = pd.to_numeric(d.judge_value, errors="coerce")
    g = d.groupby(["stimulus_id", "scale"]).v.mean().unstack("scale")
    return ((g["rounded"] - g["pointed"] + 8) / 2).to_dict()


S = sio.loadmat(f"{LACEY}/image_data.mat")["sorted_by_rating_all_P_to_R"].astype(float)  # 30 raters x 90
idxcol = np.arange(S.shape[1])
slope = np.array([np.corrcoef(S[r], idxcol)[0, 1] for r in range(S.shape[0])])
ROUND = np.where(slope > 0)[0]
POINT = np.where(slope < 0)[0]


def ceiling(corrfn, nperm=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nperm):
        r, p = rng.permutation(ROUND), rng.permutation(POINT)
        c1 = (S[r[:len(r)//2]].mean(0) - S[p[:len(p)//2]].mean(0) + 8) / 2
        c2 = (S[r[len(r)//2:]].mean(0) - S[p[len(p)//2:]].mean(0) + 8) / 2
        vals.append(corrfn(c1, c2)[0])
    rh = np.mean(vals)
    return 2 * rh / (1 + rh)


def sig(p):
    return "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "n.s."))


rows = [["Human ceiling", f"{ceiling(pearsonr):.3f}", "-", "-", f"{ceiling(spearmanr):.3f}", "-", "-", "-"]]
MODELS = [("minicpm-o-4.5", "MiniCPM-o-4.5"), ("gemini-3.5-flash", "Gemini3.5-Flash"),
          ("gemma-4-e4b", "Gemma4-E4B"), ("qwen3-omni", "Qwen3-Omni")]
res = []
for mk, mn in MODELS:
    mc = model_combine(mk)
    sh = [s for s in mc if s in hcomb]
    a = np.array([mc[s] for s in sh])
    b = np.array([hcomb[s] for s in sh])
    pr, pp = pearsonr(a, b)
    sr, sp = spearmanr(a, b)
    res.append((mn, pr, pp, sr, sp, len(sh)))
res.sort(key=lambda x: -x[1])
for mn, pr, pp, sr, sp, N in res:
    rows.append([mn, f"{pr:+.3f}", f"{pp:.2g}", sig(pp), f"{sr:+.3f}", f"{sp:.2g}", sig(sp), N])
out = pd.DataFrame(rows, columns=["Model", "pearson_r", "pearson_p", "pearson_sig",
                                  "spearman_rho", "spearman_p", "spearman_sig", "N"])
os.makedirs(OUT, exist_ok=True)
out.to_csv(f"{OUT}/exp4_combined_human_correlation.csv", index=False)
print(out.to_string(index=False))
