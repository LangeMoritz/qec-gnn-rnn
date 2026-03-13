#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-525
#SBATCH -p alvis
#SBATCH -t 1-00:00:00
#SBATCH -o logs_alvis/logs_%j.out
#SBATCH --gpus-per-node=A40:1

module purge
module load PyTorch-Geometric/2.5.0-foss-2023a-PyTorch-2.1.2-CUDA-12.1.1
source .venv/bin/activate

# sbatch run_finetune_google.sh  patch_dir  distance  load_model  [stage]  [n_epochs_s0]  [n_epochs_s1]  [n_epochs_s2]  [lr]  [note]  [wandb_project]  [basis]  [batch_size]  [n_batches]
# stage: combination of '0'(SI1000 patch circuit), '1'(p_ij DEM), '2'(real data)
# Examples:
#   sbatch run_finetune_google.sh .../d3_at_q2_7 3 <model> 012 50 100 25 1e-4 q2_7 Google-finetune Z 2048 256
#   sbatch run_finetune_google.sh .../d5_at_q4_7 5 <model> 012 50 100 25 1e-4 q4_7 Google-finetune Z 2048 128

PATCH_DIR="${1}"
DISTANCE="${2}"
LOAD_MODEL="${3}"
STAGE="${4:-012}"
N_EPOCHS_S0="${5:-50}"
N_EPOCHS_S1="${6:-100}"
N_EPOCHS_S2="${7:-25}"
LR="${8:-1e-4}"
NOTE="${9:-}"
WANDB_PROJECT="${10:-Google-finetune}"
BASIS="${11:-Z}"
BATCH_SIZE="${12:-2048}"
N_BATCHES="${13:-256}"

python -u scripts/finetune_google_patch.py \
    --patch_dir "${PATCH_DIR}" \
    --distance "${DISTANCE}" \
    --basis "${BASIS}" \
    --load_model "${LOAD_MODEL}" \
    --stage "${STAGE}" \
    --n_epochs_s0 "${N_EPOCHS_S0}" \
    --n_epochs_s1 "${N_EPOCHS_S1}" \
    --n_epochs_s2 "${N_EPOCHS_S2}" \
    --batch_size "${BATCH_SIZE}" \
    --n_batches "${N_BATCHES}" \
    --lr "${LR}" \
    ${NOTE:+--note "${NOTE}"} \
    --wandb --wandb_project "${WANDB_PROJECT}"
