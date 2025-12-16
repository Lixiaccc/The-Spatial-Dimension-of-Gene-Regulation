#!/bin/bash
#SBATCH --job-name=epif_gnn_test
#SBATCH --output=logs/epif_gnn_test_%j.out
#SBATCH --error=logs/epif_gnn_test_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
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

# Run test inference + PCC evaluation
conda run -p "${ENV}" \
  python  inference_gnn_1114.py  \
    --config ./configs/ours/finetune_ours_3000hvg.yml \
    --epif_ckpt ./experiment/finetune_ours_3000hvg_v2/ckpts/Epoch_16_Step_1952_finetune_ours_3000hvg_v2.pth \
    --gnn_ckpt ./gnn_refiner_out_1214_reg/best_gnn_refiner.pth \
    --atac_test /insomnia001/depts/msph/users/lc3716/EpiFoundation/data/ours/processed/atac_test_fixed_xy.h5ad \
    --rna_test  /insomnia001/depts/msph/users/lc3716/EpiFoundation/data/ours/processed/rna_test_binned_hvg3000_fixed_with_spatial.h5ad \
    --rna_key X_binned \
    --atac_key X \
    --x_col x --y_col y \
    --k 16 \
    --out_h5ad ./data/ours/processed/test_epif_base_vs_gnn_refined.h5ad

echo "Finished."
echo "Time: $(date)"

