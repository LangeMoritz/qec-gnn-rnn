#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-525
#SBATCH -p alvis
#SBATCH -t 7-00:00:00
#SBATCH -o logs_alvis/logs_%j.out
#SBATCH --gpus-per-node=A40:1

module purge
module load PyTorch-Geometric/2.5.0-foss-2023a-PyTorch-2.1.2-CUDA-12.1.1
source .venv/bin/activate

# auto_batch_size is on by default; pass --no_auto_batch_size explicitly if needed
# sbatch run_hierarchical.sh  base_model  d  p  t  dt  batch  nbatch  epochs  [note]  [wandb_project]  [p_list]  [test]  [trainable_base]  [random_base]  [load_path]  [lr]  [no_auto_batch_size]  [skip_mwpm_baseline]  [test_shots]  [test_rounds]
# sbatch run_hierarchical.sh  d3_p0.001_t50_dt2_260224_5979931  5  0.001  50  2  2048  244  1000  ""  GNN-iterative-decoding  "0.001 0.002 0.003 0.004 0.005"  ""  trainable_base  ""  ""  1e-4

python -u scripts/train_hierarchical.py \
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
    ${15:+--load_path "${15}"} \
    ${16:+--lr "${16}"} \
    $([[ "${17}" == "no_auto_batch_size" ]] && echo "--no_auto_batch_size") \
    $([[ "${18}" == "skip_mwpm_baseline" ]] && echo "--skip_mwpm_baseline") \
    ${19:+--test_shots "${19}"} \
    ${20:+--test_rounds ${20}}
