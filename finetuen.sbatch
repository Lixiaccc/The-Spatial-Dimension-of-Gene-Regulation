#!/bin/bash
#SBATCH --job-name=embedd
#SBATCH --output=logs/embedd_%j.out
#SBATCH --error=logs/embedd_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=5sigma1
#SBATCH --account=5sigma
mkdir -p logs

export MPLCONFIGDIR=/tmp/matplotlib-$SLURM_JOB_ID
mkdir -p $MPLCONFIGDIR

echo "Starting inference..."
echo "Time: $(date)"

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /insomnia001/depts/houlab/users/lc3716/envs/epifoundation

python extract_embeddings.py \
    --checkpoint ./experiment/finetune_ours_3000hvg_v2/ckpts/Epoch_16_Step_1952_finetune_ours_3000hvg_v2.pth \
    --output ./embeddings/train_embeddings.npz \
    --batch_size 32

echo "Done!"
echo "Time: $(date)"
