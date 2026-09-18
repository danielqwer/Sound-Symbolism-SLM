import os, sys
import pandas as pd, numpy as np, scipy.io as sio
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LACEY, RESP, OUT

m = sio.loadmat(f"{LACEY}/image_data.mat")
D = m["images_dsm"]
mr = m["means_round_ordered"]
S = m["sorted_by_rating_all_P_to_R"].astype(float)
SET = "ABCDEF"
sid = lambda i: f"mccormick_shape_{SET[i//15]}{i%15+1}"
order = [sid(int(mr[r, 1]) - 1) for r in range(90)]
iu = np.triu_indices(90, 1)
Dt = D[iu]
Drank = pd.Series(Dt).rank().values


def model_combine(mdl):
    d = pd.read_csv(f"{RESP}/{mdl}/exp4.csv")
    d = d[d.judge_ok == True].copy()
    d["v"] = pd.to_numeric(d.judge_value, errors="coerce")
    g = d.groupby(["stimulus_id", "scale"]).v.mean().unstack("scale")
    return ((g["rounded"] - g["pointed"] + 8) / 2).to_dict()


def mantel(Rt, nperm=10000, seed=123):
    rho = spearmanr(Rt, Dt)[0]
    Rrank = pd.Series(Rt).rank().values
    obs = np.corrcoef(Rrank, Drank)[0, 1]
    Rsq = np.zeros((90, 90))
    Rsq[iu] = Rrank
    Rsq[(iu[1], iu[0])] = Rrank
    rng = np.random.RandomState(seed)
    big = 1
    for _ in range(nperm):
        p = rng.permutation(90)
        pr = Rsq[np.ix_(p, p)][iu]
        if np.corrcoef(pr, Drank)[0, 1] >= obs - 1e-12:
            big += 1
    return rho, big / (nperm + 1)


def ceiling(nperm=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nperm):
        p = rng.permutation(30)
        R1 = 1 - np.corrcoef(S[p[:15]].T)
        R2 = 1 - np.corrcoef(S[p[15:]].T)
        vals.append(spearmanr(R1[iu], R2[iu])[0])
    return float(np.mean(vals))


def sig(p):
    return "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "n.s."))


print("computing human noise ceiling ...", flush=True)
rows = [["Human ceiling", f"{ceiling():.3f}", "-", "-", "-"]]
MODELS = [("minicpm-o-4.5", "MiniCPM-o-4.5"), ("gemini-3.5-flash", "Gemini3.5-Flash"),
          ("gemma-4-e4b", "Gemma4-E4B"), ("qwen3-omni", "Qwen3-Omni")]
res = []
for mk, mn in MODELS:
    mc = model_combine(mk)
    v = np.array([mc[s] for s in order])
    R = np.abs(v[:, None] - v[None, :])
    rho, p = mantel(R[iu])
    res.append((mn, rho, p))
    print(f"  {mn:16} rho={rho:+.3f} p={p:.2g}", flush=True)
res.sort(key=lambda x: -x[1])
for mn, rho, p in res:
    rows.append([mn, f"{rho:+.3f}", f"{p:.2g}", sig(p), 90])
out = pd.DataFrame(rows, columns=["Model", "spearman_rho", "mantel_p", "rho_sig", "N"])
os.makedirs(OUT, exist_ok=True)
out.to_csv(f"{OUT}/exp4_rank_agreement.csv", index=False)
print("\n" + out.to_string(index=False))
