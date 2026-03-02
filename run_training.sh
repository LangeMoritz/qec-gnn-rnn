#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-525	 	# Project
#SBATCH -p alvis 				# Partition
#SBATCH -t 3-00:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -o logs_alvis/logs_%j.out
##SBATCH --gpus-per-node=A100fat:1 # GPUs 256GB of RAM; cost factor 2.2
#SBATCH --gpus-per-node=A40:1 # GPUs 64GB of RAM; cost factor 1.0


# Load modules using pre-installed packages from the Alvis module tree
module purge
module load PyTorch-Geometric/2.5.0-foss-2023a-PyTorch-2.1.2-CUDA-12.1.1
source .venv/bin/activate

# send script
# auto_batch_size is on by default; pass --no_auto_batch_size explicitly if needed
# $13 (p_list):      optional space-separated error rates for multi-p training, e.g. "0.001 0.002 0.003 0.004 0.005"
#                    if set, overrides --p for training (--p is still used for model naming)
# $14 (noise_model): optional noise model, e.g. "SI1000" to load circuits from circuits_ZXXZ/
# $15 (test_rounds): optional space-separated round counts for evaluation, e.g. "50" or "5 10 20 50"
#                    defaults to "5 10 20 50 100 200 500 1000" if --test is set
# $16 (hidden_size):        optional GRU hidden size / final GNN output dim (default: 256)
# $17 (embedding_features): optional full GNN layer sizes, space-separated, e.g. "3 64 128 256 512"
#                           (overrides hidden_size for the GNN)
python -u scripts/train_nn.py --d "$1" --t "$2" --dt "$3" --batch_size "$4" --n_batches "$5" --n_epochs "$6" --p "$7" \
    ${8:+--intermediate} ${9:+--note "$9"} ${10:+--load_path "${10}"} ${11:+--wandb --wandb_project "${11}"} \
    ${12:+--test} ${13:+--p_list ${13}} ${14:+--noise_model "${14}"} ${15:+--test_rounds ${15}} \
    ${16:+--hidden_size "${16}"} ${17:+--embedding_features ${17}}
# sbatch run_training.sh  d  t  dt  batch  nbatch  epochs  p  [intermediate]  [note]  [load_path]  [wandb_project]  [test]  [p_list]          [noise_model]  [test_rounds]  [hidden_size]  [embedding_features]
# sbatch run_training.sh  3  50  2  2048   256     200     0.001
# sbatch run_training.sh  3  50  2  2048   256     200     0.001  ""  baseline
# sbatch run_training.sh  3  50  2  2048   256     200     0.001  int  baseline  ""  GNN-RNN-train-all-times  test
# sbatch run_training.sh  3  50  2  2048   256     200     0.001  int  baseline  ""  GNN-RNN-train-all-times  test  "0.001 0.002 0.003 0.004 0.005"
# sbatch run_training.sh  3  50  2  2048   256     200     0.001  ""   ""        ""  GNN-google-data          test  "0.001 0.005"  SI1000  ""  512  "3 64 128 256 512"