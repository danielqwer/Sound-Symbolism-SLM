import os, sys, re, warnings
import numpy as np, pandas as pd, scipy.io as sio
from scipy.stats import spearmanr, rankdata
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LACEY, RESP, OUT


def cc(x):
    while isinstance(x, np.ndarray) and x.size >= 1:
        x = x.ravel()[0]
    return str(x)


base = lambda w: re.match(r"Start-(\d+)", str(w).replace("mccormick_", "").replace("_mono", "")).group(1)

fn = sio.loadmat(f"{LACEY}/pseudowords537_Final_YJ.mat")["filenames_Final"]
ff = [base(cc(fn[i, 0])) for i in range(537)]
M = sio.loadmat(f"{LACEY}/RSA_Ordered_P_to_R_culled.mat")
combo_dsm = M["combo_Final_Order_culled_dsm"]
raters = M["combo_Final_Order_culled"]
keep = [i for i, b in enumerate(ff) if b != "253"]
order = [ff[i] for i in keep]
pos = {b: k for k, b in enumerate(order)}
H = combo_dsm[np.ix_(keep, keep)]
n = len(order)
iuH = np.triu_indices(n, 1)


def model_rdm(mdl):
    df = pd.read_csv(f"{RESP}/{mdl}/exp2.csv")
    df = df[df.judge_ok == True].copy()
    df["b"] = df.stimulus_id.map(base)
    df["jv"] = pd.to_numeric(df.judge_value, errors="coerce")
    g = df.groupby(["b", "scale"]).jv.mean().unstack("scale")
    c = (g["roundedness"] - g["pointedness"] + 8) / 2
    v = np.full(n, np.nan)
    for b, val in c.items():
        if b in pos:
            v[pos[b]] = val
    return np.abs(v[:, None] - v[None, :]), ~np.isnan(v)


def mantel(A, valid, nperm=10000, seed=123):
    idx = np.where(valid)[0]
    m = len(idx)
    iu = np.triu_indices(m, 1)
    a = A[np.ix_(idx, idx)][iu]
    b = H[np.ix_(idx, idx)][iu]
    msk = ~(np.isnan(a) | np.isnan(b))
    ar, br = rankdata(a[msk]), rankdata(b[msk])
    obs = np.corrcoef(ar, br)[0, 1]
    Br = np.zeros((m, m))
    full = rankdata(H[np.ix_(idx, idx)][iu])
    Br[iu] = full
    Br[(iu[1], iu[0])] = full
    rng = np.random.RandomState(seed)
    arc = ar - ar.mean()
    den = np.sqrt((arc**2).sum())
    big = 1
    for _ in range(nperm):
        p = rng.permutation(m)
        hp = Br[np.ix_(p, p)][iu][msk]
        hp = hp - hp.mean()
        r = (arc * hp).sum() / (den * np.sqrt((hp**2).sum()))
        if r >= obs - 1e-12:
            big += 1
    return obs, big / (nperm + 1), m


def ceiling(nperm=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nperm):
        p = rng.permutation(31)
        R1 = 1 - pd.DataFrame(raters[p[:15]][:, keep]).corr().values
        R2 = 1 - pd.DataFrame(raters[p[15:]][:, keep]).corr().values
        a, b = R1[iuH], R2[iuH]
        msk = ~(np.isnan(a) | np.isnan(b))
        vals.append(spearmanr(a[msk], b[msk])[0])
    return float(np.mean(vals))


def sig(p):
    return "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "n.s."))


MODELS = [("qwen3-omni", "Qwen3-Omni"), ("kimi-audio", "Kimi-Audio"), ("gemini-3.5-flash", "Gemini3.5-Flash"),
          ("minicpm-o-4.5", "MiniCPM-o-4.5"), ("gpt-audio-1.5", "GPT-Audio"), ("gemma-4-e4b", "Gemma4-E4B"),
          ("step-audio2", "Step-Audio2"), ("audio-flamingo3", "AudioFlamingo3")]
print("computing human noise ceiling ...", flush=True)
rows = [["Human ceiling", f"{ceiling():.3f}", "-", "-", "-"]]
res = []
for mk, mn in MODELS:
    A, val = model_rdm(mk)
    rho, p, m = mantel(A, val)
    res.append((mn, rho, p, m))
    print(f"  {mn:16} rho={rho:+.3f} p={p:.2g} N={m}", flush=True)
res.sort(key=lambda x: -x[1])
for mn, rho, p, m in res:
    rows.append([mn, f"{rho:+.3f}", f"{p:.2g}", sig(p), m])
out = pd.DataFrame(rows, columns=["Model", "spearman_rho", "mantel_p", "rho_sig", "N"])
os.makedirs(OUT, exist_ok=True)
out.to_csv(f"{OUT}/exp2_rank_agreement.csv", index=False)
print("\n" + out.to_string(index=False))
