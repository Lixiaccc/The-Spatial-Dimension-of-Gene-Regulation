#!/bin/bash
#SBATCH --job-name=spatial_gnn_reg
#SBATCH --output=logs/1213spatial_gnn_reg_%j.out
#SBATCH --error=logs/1213spatial_gnn_reg_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --qos=h012
#SBATCH -A msph
#SBATCH --gres=gpu:A6000:1

set -euo pipefail
mkdir -p logs

echo "Host: $(hostname)"
echo "Time: $(date)"
echo "JobID: ${SLURM_JOB_ID}"

module purge

ENV=/insomnia001/depts/msph/users/lc3716/envs/epifoundation_new
EPIF_ROOT=/insomnia001/depts/msph/users/lc3716/EpiFoundation

cd "${EPIF_ROOT}" || exit 1
echo "[INFO] CWD: $(pwd)"

echo "[INFO] Python:"
${ENV}/bin/python --version

conda run -p "${ENV}" \
  python "${EPIF_ROOT}/train_gnn_local_1213_v2_overfit.py" \
  --config ./configs/ours/finetune_ours_3000hvg.yml \
  --checkpoint ./experiment/finetune_ours_3000hvg_v2/ckpts/Epoch_16_Step_1952_finetune_ours_3000hvg_v2.pth \
  --rna_train ./data/ours/processed/rna_train_binned_hvg3000_fixed_with_spatial.h5ad \
  --atac_train ./data/ours/processed/atac_train_fixed_xy.h5ad \
  --rna_val ./data/ours/processed/rna_val_binned_hvg3000_fixed_with_spatial.h5ad \
  --atac_val ./data/ours/processed/atac_val_fixed_xy.h5ad \
  --rna_key X_binned \
  --atac_key X \
  --x_col x --y_col y \
  --k 16 \
  --epochs 2000 \
  --early_stop_patience 100 \
  --log_every 10 \
  --lr 5e-4 \
  --weight_decay 1e-3 \
  --dropout 0.2 \
  --edge_dropout 0.2 \
  --delta_l2 1e-3 \
  --lr_patience 20 \
  --lr_factor 0.5 \
  --min_lr 1e-6 \
  --out_dir ./gnn_refiner_out_1214_reg \
  --save_ckpt ./gnn_refiner_out_1214_reg/best_gnn_refiner.pth \
  --save_h5ad ./data/ours/processed/val_token_space_base_vs_refined_1214_reg.h5ad

echo "Finished."
echo "Time: $(date)"

