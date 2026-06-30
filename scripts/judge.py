import argparse
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

JUDGE_HF = "Qwen/Qwen3.6-35B-A3B"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def gpu_count():
    """GPU count from SLURM env, without initializing CUDA in the parent."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([x for x in cvd.split(",") if x.strip()])
    return int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))


def p_likert(t):
    m = re.search(r"\b([1-7])\b", str(t))
    return m.group(1) if m else None


def p_ab(t):
    s = str(t).upper().replace("Α", "A").replace("Β", "B").replace("Ａ", "A").replace("Ｂ", "B")
    m = re.search(r"\b([AB])\b", s)
    return m.group(1) if m else None


def p_rs(t):
    s = str(t).lower()
    r = bool(re.search(r"round", s))
    k = bool(re.search(r"spik|spike|pointed|jagged", s))
    if r and not k:
        return "rounded"
    if k and not r:
        return "spiky"
    return None


def p_rs_words(t, wr, ws):
    s = str(t)
    r = bool(wr) and bool(re.search(wr, s, re.IGNORECASE))
    k = bool(ws) and bool(re.search(ws, s, re.IGNORECASE))
    if r and not k:
        return "rounded"
    if k and not r:
        return "spiky"
    return None


PARSER = {"exp1": p_rs, "exp2": p_likert, "exp3": p_ab, "exp4": p_likert}
FMT = {
    "exp1": "the single word: rounded or spiky",
    "exp2": "a single integer from 1 to 7",
    "exp3": "the single letter: A or B",
    "exp4": "a single integer from 1 to 7",
}
LANG_NAMES = {
    "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French", "hu": "Hungarian",
    "hy": "Armenian", "it": "Italian", "ja": "Japanese", "ka": "Georgian", "ko": "Korean",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sq": "Albanian",
    "sv": "Swedish", "th": "Thai", "tr": "Turkish", "zh-cn": "Chinese", "zu": "Zulu",
}

TMPL = """The task below was given to the participant in {lang}, and the response may be in {lang}. You are extracting the answer the participant gave. Do NOT judge whether it is correct.

TASK GIVEN TO THE PARTICIPANT:
{prompt}

PARTICIPANT RESPONSE:
{raw}

Reply with ONLY the participant's chosen answer as {fmt}. If the response contains no clear answer, reply NONE."""

OUT_COLS = ["trial_id", "prompt_lang", "scale", "stimulus_id", "sample_idx",
            "option_order", "raw_answer", "regex_value", "regex_ok",
            "judge_value", "judge_ok"]


def load_trials_cols(exp):
    tri = pd.read_csv(f"{ROOT}/data/trials/{exp}.csv")
    extra = [c for c in ("phoneme", "word_rounded", "word_spiky") if c in tri]
    return tri[["trial_id", "prompt_text"] + extra]


def finish(df, model, exp, llm, tok, out_dir):
    """df must carry raw_answer + prompt_text (+ word_rounded/word_spiky for exp1)."""
    from vllm import SamplingParams

    parser = PARSER[exp]
    if exp == "exp1":
        df["regex_value"] = [p_rs_words(r, wr, ws) for r, wr, ws
                             in zip(df["raw_answer"], df["word_rounded"], df["word_spiky"])]
    else:
        df["regex_value"] = df["raw_answer"].map(parser)
    df["regex_ok"] = df["regex_value"].notna()

    prompts = [TMPL.format(lang=LANG_NAMES.get(lg, lg), prompt=p, raw=r, fmt=FMT[exp])
               for lg, p, r in zip(df["prompt_lang"], df["prompt_text"], df["raw_answer"])]
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in prompts]
    outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False)
    df["judge_value"] = [parser(o.outputs[0].text) for o in outs]
    df["judge_ok"] = df["judge_value"].notna()

    cols = OUT_COLS + (["phoneme"] if "phoneme" in df else [])
    out = f"{out_dir}/{model}/{exp}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df[cols].to_csv(out, index=False)
    print(f"[{model}/{exp}] {len(df)} rows  regex_ok={df.regex_ok.mean():.3f} "
          f"judge_ok={df.judge_ok.mean():.3f} -> {out}", flush=True)


def judge_one(model, exp, llm, tok, out_dir):
    inf = pd.read_parquet(f"{ROOT}/results/{model}/{exp}.parquet")
    df = inf.merge(load_trials_cols(exp), on="trial_id", how="left")
    df["raw_answer"] = df["raw_response"]
    finish(df, model, exp, llm, tok, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--exps", default="")
    ap.add_argument("--out-dir", default=f"{ROOT}/responses_csv")
    args = ap.parse_args()

    import torch
    from vllm import LLM
    from transformers import AutoTokenizer

    pairs = []
    for pq in sorted(glob.glob(f"{ROOT}/results/*/*.parquet")):
        model = os.path.basename(os.path.dirname(pq))
        exp = os.path.basename(pq)[:-len(".parquet")]
        if args.models and model not in args.models.split(","):
            continue
        if args.exps and exp not in args.exps.split(","):
            continue
        if exp in PARSER:
            pairs.append((model, exp))
    print("judging:", pairs, flush=True)

    tok = AutoTokenizer.from_pretrained(JUDGE_HF)
    llm = LLM(model=JUDGE_HF, tensor_parallel_size=torch.cuda.device_count(),
              gpu_memory_utilization=0.9, max_model_len=8192, trust_remote_code=True,
              enforce_eager=True)
    for model, exp in pairs:
        judge_one(model, exp, llm, tok, args.out_dir)


if __name__ == "__main__":
    main()
