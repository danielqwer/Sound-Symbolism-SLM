import argparse
import csv
import tempfile
from pathlib import Path

from prepare_data import prepare
from bk_common import DATA


def index_files(root, suffix):
    found = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix.lower() == suffix:
            key = p.stem.lower()
            if key in found:
                raise ValueError(f"Ambiguous source file: {found[key]} and {p}")
            found[key] = p.resolve()
    return found


def pseudowords(workbook, audio_dir):
    import openpyxl
    files = index_files(audio_dir, ".wav")
    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    rows = []
    try:
        sheet = book["Pseudowords"]
        values = iter(sheet.values)
        header = next(values)
        for values_row in values:
            r = dict(zip(header, values_row))
            name = str(r.get("SoundFile") or "").strip()
            if not name:
                continue
            stem = Path(name).stem if name.lower().endswith(".wav") else name

            if stem == "Start-36_mono":
                stem = "Start-36-1_mono"
            if stem.lower() not in files:
                raise FileNotFoundError(f"Missing McCormick stimulus: {stem}.wav")
            row = {"stimulus_id": "mccormick_" + stem, "audio": str(files[stem.lower()])}
            for dst, src in [("ipa_C1", "Position1_C1"), ("ipa_V1", "Position2_V1"),
                             ("ipa_C2", "Position3_C2"), ("ipa_V2", "Position4_V2")]:
                row[dst] = str(r.get(src) or "")
            rows.append(row)
    finally:
        book.close()
    if len(rows) != 537:
        raise ValueError(f"Expected 537 pseudowords in the source workbook, got {len(rows)}")
    return rows


def shapes(image_mat, shape_dir):
    import numpy as np
    from scipy.io import loadmat
    files = index_files(shape_dir, ".bmp")
    m = loadmat(image_mat)
    scores = m["sorted_by_rating_all_P_to_R"].astype(float)
    order = m["means_round_ordered"][:, 1].astype(int)
    if scores.shape != (30, 90) or sorted(order.tolist()) != list(range(1, 91)):
        raise ValueError("Unexpected image_data.mat schema")
    slopes = np.array([np.corrcoef(r, np.arange(90))[0, 1] for r in scores])
    if (slopes > 0).sum() != 17 or (slopes < 0).sum() != 13:
        raise ValueError("Unexpected human rating split (expected 17 roundedness / 13 pointedness)")
    rounded = scores[slopes > 0].mean(0)
    pointed = scores[slopes < 0].mean(0)
    rows = []
    for column, number in enumerate(order):
        name = f"{'ABCDEF'[(number - 1) // 15]}{(number - 1) % 15 + 1}"
        if name.lower() not in files:
            raise FileNotFoundError(f"Missing shape: {name}.bmp")

        rows.append({"stimulus_id": "mccormick_shape_" + name, "image": str(files[name.lower()]),
                     "roundedness_mean": round(float(rounded[column]), 3),
                     "pointedness_mean": round(float(pointed[column]), 3)})
    return rows


def write_mapping(exp, rows):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"{exp}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        count = prepare(exp, path, DATA / "stimuli" / f"{exp}.jsonl")
    print(f"{exp}: prepared {count} local stimuli")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", choices=["exp1", "exp2", "exp3", "exp4"],
                    default=["exp1", "exp2", "exp4"])
    ap.add_argument("--cwiek-dir", type=Path, default=DATA / "raw/cwiek")
    ap.add_argument("--mccormick-dir", type=Path, default=DATA / "raw/mccormick")
    ap.add_argument("--word-workbook", type=Path)
    ap.add_argument("--image-mat", type=Path, default=DATA / "human/lacey/image_data.mat")
    ap.add_argument("--rounded-shape", type=Path, help="Authorized rounded image crop for exp3")
    ap.add_argument("--spiky-shape", type=Path, help="Authorized spiky image crop for exp3")
    args = ap.parse_args()
    if "exp3" in args.exps and not (args.rounded_shape and args.spiky_shape):
        ap.error("exp3 requires --rounded-shape and --spiky-shape; see README.md")
    for exp in args.exps:
        if exp in ("exp1", "exp3"):
            rows = [{"stimulus_id": "cwiek_" + word,
                     "audio": str((args.cwiek_dir / (word + ".wav")).resolve())}
                    for word in ("bouba", "kiki")]
            if exp == "exp3":
                for r in rows:
                    r.update(rounded_shape=str(args.rounded_shape.resolve()),
                             spiky_shape=str(args.spiky_shape.resolve()))
        elif exp == "exp2":
            rows = pseudowords(args.word_workbook or args.mccormick_dir / "SoS Pseudoword Database.xlsx",
                               args.mccormick_dir / "Pseudowords")
        else:
            rows = shapes(args.image_mat, args.mccormick_dir / "Shapes")
        write_mapping(exp, rows)


if __name__ == "__main__":
    main()
