import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_file(path):

    if not path.is_file() or not path.stat().st_size:
        return False
    with path.open("rb") as fh:
        head = fh.read(512)
    if b"<html" in head.lower() or b"<!doctype html" in head.lower():
        return False
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    if suffix == ".bmp":
        return head[:2] == b"BM"
    if suffix == ".xlsx":
        return head[:2] == b"PK"
    if suffix == ".mat":
        return head.startswith(b"MATLAB") or head.startswith(b"\x89HDF")
    return True


def download(url, dest, retries=4):
    dest = Path(dest)
    if validate_file(dest):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".download" + dest.suffix)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as fh:
                while chunk := response.read(1024 * 1024):
                    fh.write(chunk)
            if not validate_file(tmp):
                raise ValueError(f"Invalid download: {url}")
            tmp.replace(dest)
            return
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", choices=["human", "cwiek", "mccormick", "acoustic"],
                    default=["human"])
    ap.add_argument("--out-dir", type=Path,
                    default=Path(os.environ.get("BK_DATA_DIR") or ROOT.parent / "sound-symbolism-data").expanduser())
    ap.add_argument("--list", action="store_true", help="List selected files without downloading")
    args = ap.parse_args()
    catalog = json.loads((ROOT / "configs/data_sources.json").read_text())
    files = [r for r in catalog["files"] if r["group"] in args.groups]
    failures = []
    for row in files:
        print(f"{row['path']} <- {row['url']}", flush=True)
        if not args.list:
            try:
                download(row["url"], args.out_dir / row["path"])
            except Exception as exc:
                failures.append(row["path"])
                print(f"FAILED: {exc}", flush=True)
    if failures:
        raise SystemExit(f"Failed to download {len(failures)} files. Rerun to resume: {failures}")
    print(f"{'Listed' if args.list else 'Ready'}: {len(files)} files")


if __name__ == "__main__":
    main()
