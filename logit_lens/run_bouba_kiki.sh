#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Bouba-Kiki logit lens — shell entrypoint (called by SLURM via srun)
#
# Override at submission:
#   sbatch --export=ALL,BACKEND=minicpm,MODE=test_image   run_bouba_kiki.job
#   sbatch --export=ALL,BACKEND=qwen3-omni,MODE=test_audio run_bouba_kiki.job
#   sbatch --export=ALL,BACKEND=qwen3-omni,MODE=full_image run_bouba_kiki.job
#   sbatch --export=ALL,BACKEND=minicpm,MODE=full_audio    run_bouba_kiki.job
#
# BACKEND values : minicpm | qwen3-omni   (default: minicpm)
# MODE values    : test_image | test_audio | full_image | full_audio
# ─────────────────────────────────────────────────────────────────────────────

export PYTHONIOENCODING=UTF-8
export HF_HOME="/work/wilzwork23/hf_cache"

# ── BACKEND → model path + conda env ─────────────────────────────────────────
BACKEND="${BACKEND:-minicpm}"

case "${BACKEND}" in
    minicpm)
        MODEL_PATH="/work/wilzwork23/models/MiniCPM-o-4_5"
        CONDA_ENV="minicpm_logit_lens"
        ;;
    qwen3-omni)
        MODEL_PATH="/work/wilzwork23/models/Qwen3-Omni-30B-A3B-Instruct"
        CONDA_ENV="qwen3_omni_cap"
        ;;
    *)
        echo "ERROR: Unknown BACKEND='${BACKEND}'. Choose: minicpm | qwen3-omni"
        exit 1
        ;;
esac

# ── FIXED CONFIG ─────────────────────────────────────────────────────────────
DTYPE="${DTYPE:-bf16}"
BOBA_LANG="${BOBA_LANG:-en}"
PROJ_DIR="/home/wilzwork23/boba_kiki_logit_lens"
OUTPUT_DIR="${PROJ_DIR}/results"
LOG_DIR="${PROJ_DIR}/logs"
# ─────────────────────────────────────────────────────────────────────────────

# ── MODE → EXP + SAMPLE ──────────────────────────────────────────────────────
MODE="${MODE:-full_image}"

case "${MODE}" in
    test_image)
        EXP="exp5"
        SAMPLE=20
        BOBA_LANG="en"
        ;;
    test_audio)
        EXP="exp2"
        SAMPLE=20
        BOBA_LANG="en"
        ;;
    full_image)
        EXP="exp5"
        SAMPLE=0
        ;;
    full_audio)
        EXP="exp2"
        SAMPLE=0
        ;;
    *)
        echo "ERROR: Unknown MODE='${MODE}'. Choose: test_image | test_audio | full_image | full_audio"
        exit 1
        ;;
esac
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

source /home/wilzwork23/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# ── Merge mode ────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--merge" ]]; then
    echo "=========================================="
    echo "MERGE phase"
    echo "Backend   : ${BACKEND}"
    echo "Exp       : ${EXP}"
    echo "Lang      : ${BOBA_LANG}"
    echo "Output    : ${OUTPUT_DIR}"
    echo "Start     : $(date)"
    echo "=========================================="

    python "${PROJ_DIR}/run_bouba_kiki_lens.py" \
        --mode    merge          \
        --backend "${BACKEND}"   \
        --exp     "${EXP}"       \
        --lang    "${BOBA_LANG}" \
        --output  "${OUTPUT_DIR}"

    echo "=========================================="
    echo "Merge done : $(date)"
    echo "=========================================="
    exit 0
fi

# ── Worker mode ───────────────────────────────────────────────────────────────
WORKER_ID="${SLURM_PROCID:-0}"
NUM_WORKERS="${SLURM_NTASKS:-1}"
LOCAL_GPU="${SLURM_LOCALID:-0}"

export CUDA_VISIBLE_DEVICES="${LOCAL_GPU}"

echo "=========================================="
echo "Node      : $(hostname)"
echo "Backend   : ${BACKEND}"
echo "Model     : ${MODEL_PATH}"
echo "Conda env : ${CONDA_ENV}"
echo "Worker    : ${WORKER_ID}/${NUM_WORKERS}  (GPU ${LOCAL_GPU})"
echo "Mode      : ${MODE}"
echo "Exp       : ${EXP}"
echo "Sample    : ${SAMPLE:-all}"
echo "Lang      : ${BOBA_LANG}"
echo "Dtype     : ${DTYPE}"
echo "Start     : $(date)"
echo "=========================================="

echo "Python    : $(which python)"
echo "PyTorch   : $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo ""

python "${PROJ_DIR}/run_bouba_kiki_lens.py" \
    --mode        worker           \
    --backend     "${BACKEND}"     \
    --model       "${MODEL_PATH}"  \
    --exp         "${EXP}"         \
    --lang        "${BOBA_LANG}"   \
    --sample      "${SAMPLE}"      \
    --dtype       "${DTYPE}"       \
    --device      "cuda:0"         \
    --num_workers "${NUM_WORKERS}" \
    --worker_id   "${WORKER_ID}"   \
    --output      "${OUTPUT_DIR}"

echo ""
echo "=========================================="
echo "Worker ${WORKER_ID} done : $(date)"
echo "=========================================="
