#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-525        # Project
#SBATCH -p alvis                  # Partition
#SBATCH -t 3-00:00:00             # time limit days-hours:minutes:seconds
#SBATCH -o logs_alvis/logs_%j.out
#SBATCH --gpus-per-node=A40:1     # GPUs 64GB of RAM; cost factor 1.0

module purge
module load PyTorch-Geometric/2.5.0-foss-2023a-PyTorch-2.1.2-CUDA-12.1.1
source .venv/bin/activate

# Ensure ldpc is available for BP-OSD baseline
pip install ldpc --quiet 2>/dev/null

# Usage:
#   sbatch run_bb_training.sh  code_size  t  p  epochs  [wandb_project]  [load]  [p_list]  [hidden]  [embed]  [lr]  [dt]  [test]  [n_gru]  [decoder_hidden]
#
# $1:  code_size        72|90|108|144|288  (default: 72)
# $2:  t                syndrome rounds    (default: code distance)
# $3:  p                error rate         (default: 0.001)
# $4:  epochs           training epochs    (default: 500)
# $5:  wandb_project    enables --wandb if set (e.g. "GNN-RNN-BB-codes")
# $6:  load             model name to resume (no models/ prefix, no .pt)
# $7:  p_list           space-separated multi-p training (e.g. "0.001 0.003 0.005")
# $8:  hidden           GRU hidden size (default: 256)
# $9:  embed            space-separated GNN layer sizes (e.g. "4 64 128 256 512 1024")
# $10: lr               learning rate (default: 1e-3; min_lr auto-set to same value)
# $11: dt               sliding window size; g_max = t - dt + 2 (default: 2)
# $12: test             any non-empty string enables --test (e.g. "test")
# $13: n_gru            number of GRU layers (default: 4)
# $14: decoder_hidden   MLP head hidden dim; omit for linear head (e.g. 256)
#
# Examples:
#   sbatch run_bb_training.sh 72 6 0.001 500 GNN-RNN-BB-codes
#   sbatch run_bb_training.sh 72 6 0.001 500 GNN-RNN-BB-codes "" "0.001 0.003 0.005"
#   sbatch run_bb_training.sh 72 6 0.001 300 GNN-RNN-BB-codes my_model
#   sbatch run_bb_training.sh 72 6 0.001 1000 GNN-RNN-BB-codes "" "" 256 "4 64 128 256 512 1024"
#   sbatch run_bb_training.sh 72 6 0.001 1000 GNN-RNN-BB-codes "" "0.001 0.002 0.003" 1024 "4 64 128 256 512 1024" 1e-3 2 test 4 256

python -u scripts/train_bb.py \
    --code_size "${1:-72}" \
    --t         "${2:-6}"  \
    --p         "${3:-0.001}" \
    --epochs    "${4:-500}" \
    --batch     4096 \
    --nbatch    128 \
    ${5:+--wandb --wandb_project "$5"} \
    ${6:+--load "$6"} \
    ${7:+--p_list $7} \
    ${8:+--hidden "$8"} \
    ${9:+--embed $9} \
    ${10:+--lr "${10}"} \
    ${11:+--dt "${11}"} \
    ${12:+--test} \
    ${13:+--n_gru "${13}"} \
    ${14:+--decoder_hidden "${14}"}
