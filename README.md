# Hearing Like Humans? Sound Symbolism and Perceptual Alignment in Speech Language Models

Code for the paper's four behavioral experiments: auditory classification, auditory ratings, sound–shape matching, and visual ratings. [Citation](CITATION.cff).

![Experimental design](assets/experimental-design.png)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
export BK_DATA_DIR="$PWD/../sound-symbolism-data"
```

Set API keys in `.env`. For local models, install the checkpoint's upstream runtime separately; model IDs and adapters are listed in [scripts/config.py](scripts/config.py). The response judge requires Qwen3.6, vLLM, and a GPU.

## Data

Third-party data is not included. Download from [Ćwiek et al.](https://osf.io/w7crs/) (bouba/kiki audio), [McCormick et al.](https://osf.io/ekpgh/) (pseudowords and shapes), and [Lacey et al.](https://osf.io/y9zjc/) (human ratings and dissimilarities), following their terms. [configs/data_sources.json](configs/data_sources.json) contains download links only. Data stays outside this repository in `BK_DATA_DIR`.

```bash
python scripts/fetch_data.py --groups human cwiek mccormick
python scripts/prepare_sources.py --exps exp1 exp2 exp4
```

Experiment 3 also needs the rounded and spiky shape crops from Ćwiek et al.'s Figure 1; supply local copies matching the original experiment:

```bash
python scripts/prepare_sources.py --exps exp3 \
  --rounded-shape /path/to/rounded.png --spiky-shape /path/to/spiky.png
python scripts/build_trials.py
python scripts/validate_data.py --check-media
```

Preparation writes local media manifests to `stimuli/` and prompts/seeds to `trials/` under `BK_DATA_DIR`. Expected trial counts per model are 1,500 / 32,220 / 1,500 / 5,400. Experiment 2 runs 537 words and excludes `Start-253` during analysis; Experiment 4 retains the archive's `e5_` trial IDs.

## Run

```bash
python scripts/run_experiment.py --model qwen3-omni --exp exp1 \
  --out results/qwen3-omni/exp1.parquet
python scripts/judge.py --models qwen3-omni --exps exp1 --out-dir responses_csv
```

Repeat for compatible model/experiment pairs in `scripts/config.py`. Inference resumes by trial ID; use `--lang en --limit 4` with a separate output path for a quick check. All experimental prompts are in [prompts/prompts.json](prompts/prompts.json).

After generating and judging all required model responses, run the analysis scripts:

```bash
python analysis/t1_congruence.py
python analysis/exp1_significance.py
python analysis/exp3_congruence.py
python analysis/order_effect.py
python analysis/t2_correlation.py
python analysis/t2_rank_agreement.py
python scripts/fetch_data.py --groups acoustic
python analysis/spectral_dsm.py
python analysis/t3_acoustic_rsa.py
python analysis/exp4_correlation.py
python analysis/exp4_rank_agreement.py
```

Tables are written to `results/tables/`. The paper's logit-lens analysis is not included in this release.
