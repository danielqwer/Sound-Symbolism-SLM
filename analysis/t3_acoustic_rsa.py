import os, sys, re, warnings
import numpy as np, pandas as pd, scipy.io as sio
from scipy.stats import spearmanr, rankdata
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LACEY, ACOUSTIC, RESP, OUT


def cell(x):
    while isinstance(x, np.ndarray) and x.size >= 1:
        x = x.ravel()[0]
    return str(x)


def base(w):
    return re.match(r"Start-(\d+)", str(w).replace("mccormick_", "").replace("_mono", "")).group(1)


fn = sio.loadmat(f"{LACEY}/pseudowords537_Final_YJ.mat")["filenames_Final"]
ff = [base(cell(fn[i, 0])) for i in range(537)]
combo_dsm = sio.loadmat(f"{LACEY}/RSA_Ordered_P_to_R_culled.mat")["combo_Final_Order_culled_dsm"]

P = {}
for lab, f in [("spectral_tilt", "dsm_spectral_tilt.npy"), ("temporal_FFT", "dsm_freq_FFT.npy"),
               ("speech_envelope", "dsm_speech_envelope.npy")]:
    P[lab] = np.load(f"{ACOUSTIC}/{f}")
xl = pd.read_excel(f"{ACOUSTIC}/voiceReportData.xlsx", sheet_name="Sheet1")
vcols = {"autocorrelation": "Autocorr", "HNR": "HNR", "fraction_unvoiced": "FUF", "pulse_number": "Pulse #",
         "jitter": "Jitter", "shimmer": "Shimmer", "pitch_SD": "Pitch std dev"}
for lab, col in vcols.items():
    v = xl[col].values[::-1].astype(float)
    P[lab] = np.abs(v[:, None] - v[None, :])
PARAMS = ["spectral_tilt", "temporal_FFT", "speech_envelope", "autocorrelation", "HNR",
          "fraction_unvoiced", "pulse_number", "jitter", "shimmer", "pitch_SD"]

keep = [i for i, b in enumerate(ff) if b != "253"]
order = [ff[i] for i in keep]
pos = {b: k for k, b in enumerate(order)}
combo_dsm = combo_dsm[np.ix_(keep, keep)]
for k in P:
    P[k] = P[k][np.ix_(keep, keep)]
n = len(order)


def model_rdm(mdl):
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
    return np.abs(v[:, None] - v[None, :]), ~np.isnan(v)


def rsa_mantel(A, B, valid=None, nperm=10000, seed=123):
    idx = np.where(valid)[0] if valid is not None else np.arange(A.shape[0])
    m = len(idx)
    iu = np.triu_indices(m, 1)
    a = A[np.ix_(idx, idx)][iu]
    b = B[np.ix_(idx, idx)][iu]
    msk = ~(np.isnan(a) | np.isnan(b))
    a, b = a[msk], b[msk]
    if len(a) < 10:
        return np.nan, np.nan, m
    ma = rankdata(a)
    rho = float(spearmanr(a, b)[0])
    RH = np.zeros((m, m))
    rr = rankdata(B[np.ix_(idx, idx)][iu])
    RH[iu] = rr
    RH[(iu[1], iu[0])] = rr
    mac = ma - ma.mean()
    den = np.sqrt((mac**2).sum())
    rng = np.random.RandomState(seed)
    big = 0
    for _ in range(nperm):
        p = rng.permutation(m)
        hp = RH[np.ix_(p, p)][iu][msk]
        r = float((mac * (hp - hp.mean())).sum() / (den * np.sqrt(((hp - hp.mean())**2).sum())))
        if r >= rho:
            big += 1
    return rho, (big + 1) / (nperm + 1), m


def sig(p):
    return "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "ns"))


iu = np.triu_indices(n, 1)
human = {}
for k in PARAMS:
    a, b = combo_dsm[iu], P[k][iu]
    msk = ~(np.isnan(a) | np.isnan(b))
    human[k] = spearmanr(a[msk], b[msk])[0]

MODELS = [("qwen3-omni", "Qwen3-Omni"), ("kimi-audio", "Kimi-Audio"), ("minicpm-o-4.5", "MiniCPM-o-4.5"),
          ("gpt-audio-1.5", "GPT-Audio"), ("step-audio2", "Step-Audio2"), ("gemma-4-e4b", "Gemma4-E4B"),
          ("audio-flamingo3", "AudioFlamingo3"), ("gemini-3.5-flash", "Gemini3.5-Flash")]
rows = [["Human"] + [f"{human[k]:+.3f}" for k in PARAMS]]
for mdl, name in MODELS:
    A, valid = model_rdm(mdl)
    cells = []
    for k in PARAMS:
        rho, p, m = rsa_mantel(A, P[k], valid)
        cells.append(f"{rho:+.3f}{sig(p)}")
    rows.append([name] + cells)
    print(f"{name:16s} " + " ".join(cells), flush=True)
out = pd.DataFrame(rows, columns=["Model"] + PARAMS)
os.makedirs(OUT, exist_ok=True)
out.to_csv(f"{OUT}/exp2_acoustic_rsa.csv", index=False)
