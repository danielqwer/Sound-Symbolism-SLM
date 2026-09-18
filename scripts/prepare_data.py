import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from bk_common import DATA

MEDIA = {"exp1": ("audio",), "exp2": ("audio",),
         "exp3": ("audio", "rounded_shape", "spiky_shape"), "exp4": ("image",)}

def prepare(exp, mapping, output):
    mapping, output = Path(mapping).resolve(), Path(output).resolve()
    rows, seen = [], set()
    with mapping.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"stimulus_id", *MEDIA[exp]}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for number, entry in enumerate(reader, 2):
            row = {k: v.strip() for k, v in entry.items() if k and v and v.strip()}
            sid = row.get("stimulus_id")
            if not sid or sid in seen:
                raise ValueError(f"Row {number}: empty or duplicate stimulus_id")
            seen.add(sid)
            for field in MEDIA[exp]:
                if field not in row:
                    raise ValueError(f"Row {number}: missing {field}")
                path = (mapping.parent / row[field]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Row {number}, {field}: {path}")
                if field == "audio":
                    import soundfile as sf
                    info = sf.info(path)
                    if info.frames <= 0 or info.samplerate <= 0:
                        raise ValueError(f"Empty or invalid audio: {path}")
                else:
                    from PIL import Image
                    with Image.open(path) as image:
                        image.verify()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                row[field] = {"path": str(path), "sha256": digest}
            for field in ("roundedness_mean", "pointedness_mean"):
                if field in row:
                    row[field] = float(row[field])
                    if not math.isfinite(row[field]) or not 1 <= row[field] <= 7:
                        raise ValueError(f"{sid}: {field} must be in [1, 7]")
            rows.append(row)
    if not rows:
        raise ValueError("The mapping contains no stimuli")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(output)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=MEDIA)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or DATA / "stimuli" / f"{args.exp}.jsonl"
    count = prepare(args.exp, args.mapping, out)
    print(f"Validated {count} stimuli -> {out}")


if __name__ == "__main__":
    main()
