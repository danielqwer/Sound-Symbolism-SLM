import os, sys
import numpy as np, scipy.io as sio, scipy.signal as sg, soundfile as sf
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LACEY, ACOUSTIC


def cell(x):
    while isinstance(x, np.ndarray) and x.size >= 1:
        x = x.ravel()[0]
    return str(x)


fn = sio.loadmat(f"{LACEY}/pseudowords537_Final_YJ.mat")["filenames_Final"]
names = [cell(fn[i, 0]) for i in range(fn.shape[0])]
Fs, TARGET = 22050, 10077


def load_resamp(name):
    y, sr = sf.read(f"{ACOUSTIC}/wavs/{name}.wav")
    if y.ndim > 1:
        y = y.mean(1)
    if len(y) != TARGET:
        y = sg.resample(y, TARGET)
    return y.astype(np.float64)


def tilt_vec(y):
    N = len(y)
    ydft = np.fft.fft(y)[1:N // 2 + 1]
    psdy = (1 / (Fs * N)) * np.abs(ydft)**2
    psdy[1:-1] *= 2
    freq = np.arange(0, Fs / 2 + 1e-9, Fs / len(y))
    L = min(len(psdy), len(freq))
    psdy, freq = psdy[:L], freq[:L]
    dB = 10 * np.log10(psdy + 1e-30)
    window, overlap = 600, 15
    step = window - overlap
    slopes, p = [], 0
    while p + window <= L:
        slopes.append(np.polyfit(freq[p:p + window], dB[p:p + window], 1)[0])
        p += step
    return np.array(slopes)


def fft_vec(y):
    f, t, Z = sg.stft(y, fs=Fs, window=sg.get_window("hamming", 100, fftbins=True),
                      nperseg=100, noverlap=80, nfft=100, boundary=None, padded=False, return_onesided=True)
    A = np.abs(np.log2(Z[:11, :].astype(np.complex128) + 0j))
    return A.flatten(order="F")


def env_vec(y):
    w = 120
    k = np.ones(w) / w
    up = np.sqrt(np.convolve(y**2, k, mode="same"))
    return np.concatenate([up, -up])


def build_dsm(vectors):
    M = np.column_stack(vectors)
    D = 1 - np.corrcoef(M, rowvar=False)
    np.fill_diagonal(D, 0)
    return D


print("loading + computing features ...", flush=True)
Y = [load_resamp(n) for n in names]
cm = sio.loadmat(f"{LACEY}/RSA_Ordered_P_to_R_culled.mat")["combo_Final_Order_culled_dsm"]
iu = np.triu_indices(537, 1)
for label, fnc in [("spectral_tilt", tilt_vec), ("freq_FFT", fft_vec), ("speech_envelope", env_vec)]:
    vecs = [fnc(y) for y in Y]
    L = min(len(v) for v in vecs)
    D = build_dsm([v[:L] for v in vecs])
    np.save(f"{ACOUSTIC}/dsm_{label}.npy", D)
    rho, _ = spearmanr(D[iu], cm[iu])
    print(f"{label:16s} featlen={L:5d}  human r (Spearman vs combo_dsm) = {rho:+.3f}", flush=True)
