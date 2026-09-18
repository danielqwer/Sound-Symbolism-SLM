import argparse

import bk_common as bk


def validate(exp, check_media=False):
    stimuli = bk.load_stimuli(exp)
    trials = bk.load_trials(exp)
    if not trials:
        raise ValueError(f"{exp}: empty trial table")
    seen = set()
    for t in trials:
        tid = t.get("trial_id")
        if not tid or tid in seen:
            raise ValueError(f"{exp}: empty or duplicate trial_id: {tid}")
        seen.add(tid)
        if t.get("stimulus_id") not in stimuli:
            raise ValueError(f"{tid}: unknown stimulus {t.get('stimulus_id')}")
        if not t.get("prompt_text") or not t.get("prompt_lang"):
            raise ValueError(f"{tid}: missing prompt text or language")
        for field in ("sample_idx", "trial_seed"):
            value = t.get(field)
            if value is None or int(value) != float(value) or int(value) < 0:
                raise ValueError(f"{tid}: invalid {field}")
        if exp in ("exp1", "exp3"):
            if t.get("option_order") not in ("rounded_first", "spiky_first"):
                raise ValueError(f"{tid}: invalid option_order")
        else:
            scales = ("roundedness", "pointedness") if exp == "exp2" else ("rounded", "pointed")
            if t.get("scale") not in scales:
                raise ValueError(f"{tid}: invalid scale")
        if exp == "exp1" and not (t.get("word_rounded") and t.get("word_spiky")):
            raise ValueError(f"{tid}: missing answer vocabulary")
    if exp == "exp4":
        for sid, s in stimuli.items():
            for field in ("roundedness_mean", "pointedness_mean"):
                if field not in s or not 1 <= float(s[field]) <= 7:
                    raise ValueError(f"{sid}: missing or invalid {field}")
    for sid, s in stimuli.items():
        for field, media in s.items():
            if isinstance(media, dict) and "path" in media:
                if check_media:
                    if field == "audio":
                        audio, sr = bk.decode_audio(media)
                        if not len(audio) or sr <= 0:
                            raise ValueError(f"{sid}: invalid audio")
                    else:
                        bk.decode_image(media)
    return len(stimuli), len(trials)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", choices=["exp1", "exp2", "exp3", "exp4"],
                    default=["exp1", "exp2", "exp3", "exp4"])
    ap.add_argument("--check-media", action="store_true")
    args = ap.parse_args()
    for exp in args.exps:
        ns, nt = validate(exp, args.check_media)
        print(f"{exp}: OK ({ns} stimuli, {nt} trials)")


if __name__ == "__main__":
    main()
