#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-525
#SBATCH -p alvis
#SBATCH -t 3-00:00:00
#SBATCH -o logs_alvis/logs_%j.out
#SBATCH --gpus-per-node=A40:1

module purge
module load PyTorch-Geometric/2.5.0-foss-2023a-PyTorch-2.1.2-CUDA-12.1.1
source .venv/bin/activate
pip install -q line_profiler

# Same args as run_hierarchical.sh — runs under kernprof for line-by-line timing.
# Profile output saved to logs_alvis/profile_<JOBID>.lprof
# sbatch run_hierarchical_profile.sh  base_model  d  p  t  dt  batch  nbatch  epochs  [note]  [wandb_project]  [p_list]  [test]  [trainable_base]  [random_base]  [load_path]

LPROF_OUT="logs_alvis/profile_${SLURM_JOB_ID}.lprof"

kernprof -l -o "$LPROF_OUT" scripts/train_hierarchical.py -- \
    --base_model "$1" \
    --d "$2" \
    --p "$3" \
    --t "$4" \
    --dt "$5" \
    --batch_size "$6" \
    --n_batches "$7" \
    --n_epochs "$8" \
    ${9:+--note "$9"} \
    ${10:+--wandb --wandb_project "${10}"} \
    ${11:+--p_list ${11}} \
    $([[ "${12}" == "test" ]] && echo "--test") \
    $([[ "${13}" == "trainable_base" ]] && echo "--trainable_base") \
    $([[ "${14}" == "random_base" ]] && echo "--random_base") \
    ${15:+--load_path "${15}"}

python -m line_profiler "$LPROF_OUT"
