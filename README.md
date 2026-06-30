# Hearing Like Humans? Sound Symbolism and Perceptual Alignment in Speech Language Models

## Models

`qwen3-omni`, `minicpm-o-4.5`, `gemma-4-e4b`, `audio-flamingo3`, `kimi-audio`,
`step-audio2`, `gpt-audio-1.5`, `gemini-3.5-flash`. The HF repo / API id for each is
in `scripts/config.py`. Set up an environment for each local model as its upstream
requires; API models read keys from `.env`.

## Data

The stimuli and trial grids are **not yet included** — we will release them as soon as
possible. Model responses are produced by running the pipeline, not distributed. Data
locations are configurable in `analysis/paths.py`.

The third-party human ratings used in the analysis are public:
- Lacey et al. 2020 — [OSF y9zjc](https://osf.io/y9zjc/): `RSA_Ordered_P_to_R_culled.mat`,
  `pseudowords537_Final_YJ.mat`, `image_data.mat`, pseudoword `wavs/`, `voiceReportData.xlsx`.
- Ćwiek et al. 2022 — [OSF w7crs](https://osf.io/w7crs/): `web_by_trial.csv` (auditory web experiment).

`scripts/fetch_data.py` downloads these OSF resources into `data/human/`.

## Setup

```bash
cp .env.example .env        # then fill in OPENAI_API_KEY / GEMINI_API_KEY
pip install numpy pandas scipy soundfile openpyxl pyarrow
```

## Run

```bash
# 1. build trial files from the downloaded grids
python scripts/build_trials.py

# 2. inference — one (model, exp)
python scripts/run_experiment.py --model qwen3-omni --exp exp1 \
    --out results/qwen3-omni/exp1.parquet

# 3. LLM judge: parse each raw response into the scored value (needs a GPU)
python scripts/judge.py --models qwen3-omni --exps exp1 --out-dir responses_csv

# 4. tables
python analysis/t1_congruence.py        # exp1 congruence
python analysis/exp1_significance.py    # exp1 binomial (vs chance) + bouba-vs-kiki Wilcoxon
python analysis/order_effect.py         # exp1 & exp3 position/order effect
python analysis/t2_correlation.py       # exp2 first-order
python analysis/t2_rank_agreement.py    # exp2 second-order RSA
python analysis/t3_acoustic_rsa.py      # exp2 acoustic RSA
python analysis/exp4_correlation.py     # exp4 first-order
python analysis/exp4_rank_agreement.py  # exp4 second-order RSA
```
