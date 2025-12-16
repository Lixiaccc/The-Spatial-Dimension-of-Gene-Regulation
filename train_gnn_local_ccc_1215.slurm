#!/bin/bash
#SBATCH --job-name=spatial_gnn_debug
#SBATCH --output=logs/1213spatial_gnn_%j.out
#SBATCH --error=logs/1213spatial_gnn_%j.err
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

EPIF_CKPT=${EPIF_ROOT}/experiment/finetune_ours_3000hvg_v2/ckpts/Epoch_16_Step_1952_finetune_ours_3000hvg_v2.pth

RNA_VOCAB=${EPIF_ROOT}/rna_vocab_ours_canonical.json
ATAC_VOCAB=${EPIF_ROOT}/atac_vocab_ours.json
CELL_VOCAB=${EPIF_ROOT}/cell_full.json
BATCH_VOCAB=${EPIF_ROOT}/batch_full.json
CHR_VOCAB=${EPIF_ROOT}/chr_vocab_ours.json
GENE2CHR=${EPIF_ROOT}/gene2chr_ours_canonical.json

RNA_TRAIN=${EPIF_ROOT}/data/ours/processed/rna_train_binned_hvg3000_fixed_with_spatial.h5ad
ATAC_TRAIN=${EPIF_ROOT}/data/ours/processed/atac_train_fixed_xy.h5ad
RNA_VAL=${EPIF_ROOT}/data/ours/processed/rna_val_binned_hvg3000_fixed_with_spatial.h5ad
ATAC_VAL=${EPIF_ROOT}/data/ours/processed/atac_val_fixed_xy.h5ad

echo "[INFO] Python:"
${ENV}/bin/python --version

conda run -p "${ENV}" \
  python "${EPIF_ROOT}/train_gnn_local_ccc_1215.py" \
  --config ./configs/ours/finetune_ours_3000hvg.yml \
  --checkpoint ./experiment/finetune_ours_3000hvg_v2/ckpts/Epoch_16_Step_1952_finetune_ours_3000hvg_v2.pth \
  --rna_train ./data/ours/processed/rna_train_binned_hvg3000_fixed_with_spatial.h5ad \
  --atac_train ./data/ours/processed/atac_train_fixed_xy_with_rna_ccc.h5ad \
  --rna_val ./data/ours/processed/rna_val_binned_hvg3000_fixed_with_spatial.h5ad \
  --atac_val ./data/ours/processed/atac_val_fixed_xy_with_rna_ccc.h5ad \
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
  --out_dir ./gnn_refiner_out_ccc_attn \
  --save_ckpt ./gnn_refiner_out_ccc_attn/best_gnn_refiner.pth \
  --save_h5ad ./data/ours/processed/val_gene_space_base_vs_refined_ccc_attn.h5a
decho "Finished."
echo "Time: $(date)"

