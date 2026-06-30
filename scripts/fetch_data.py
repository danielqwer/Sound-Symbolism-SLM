# Download the third-party human reference data (OSF) into data/human/.
import json
import os
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUMAN = os.path.join(ROOT, "data", "human")
LACEY = os.path.join(HUMAN, "lacey")
ACOUSTIC = os.path.join(HUMAN, "lacey_acoustic")
WAVS = os.path.join(ACOUSTIC, "wavs")
CWIEK = os.path.join(HUMAN, "cwiek")

OSF_LACEY = "y9zjc"
OSF_CWIEK = "w7crs"
LACEY_MATS = {"RSA_Ordered_P_to_R_culled.mat", "pseudowords537_Final_YJ.mat", "image_data.mat"}


def _get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def osf_files(node):
    """Yield (name, download_url) for every file in an OSF project (recursive)."""
    stack = [f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/?page%5Bsize%5D=100"]
    while stack:
        page = _get_json(stack.pop())
        for x in page["data"]:
            a = x["attributes"]
            if a["kind"] == "folder":
                stack.append(x["relationships"]["files"]["links"]["related"]["href"])
            else:
                yield a["name"], x["links"]["download"]
        nxt = page["links"].get("next")
        if nxt:
            stack.append(nxt)


def download(url, dest, is_wav=False, retries=4):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            if is_wav and data[:4] != b"RIFF":          # OSF rate-limit returns an HTML page
                raise ValueError("not a WAV (rate-limited?)")
            with open(dest, "wb") as f:
                f.write(data)
            return
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAILED {os.path.basename(dest)}: {e}")
            else:
                time.sleep(2 * (attempt + 1))


def fetch_lacey():
    print("Lacey (OSF y9zjc) ...")
    for name, url in osf_files(OSF_LACEY):
        if name in LACEY_MATS:
            download(url, os.path.join(LACEY, name))
        elif name == "voiceReportData.xlsx":
            download(url, os.path.join(ACOUSTIC, name))
        elif name.lower().endswith(".wav"):
            download(url, os.path.join(WAVS, name), is_wav=True)


def fetch_cwiek():
    print("Cwiek (OSF w7crs) ...")
    for name, url in osf_files(OSF_CWIEK):
        if name == "web_by_trial.csv":
            download(url, os.path.join(CWIEK, name))


if __name__ == "__main__":
    fetch_lacey()
    fetch_cwiek()
    print("done.")
