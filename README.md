EpiFoundation Spatial Refinement (ATAC → RNA with GNN + CCC)

This repository contains the code for our course/research project:

“The Spatial Dimension of Gene Regulation: Integrating ATAC-seq with Tissue Context Using a Graph Neural Network for Gene Expression Predictions.”

We predict gene expression (RNA) from chromatin accessibility (ATAC) using a two-stage pipeline:

Base model (EpiFoundation fine-tuning): produces per-cell gene expression predictions and a cell embedding.

Spatial refinement: uses either

a Spatial kNN GNN (train_gnn_local_1213_v2_overfit.py), or

a CCC-enhanced GNN with LIANA-derived ligand–receptor signals (train_gnn_local_ccc_1215.py)
to refine base predictions using tissue neighborhood information.


Repository Structure

Key scripts in this repo:

Core training scripts

finetune_ours.py
Fine-tunes EpiFoundation on paired spatial ATAC-RNA data (base model).

train_gnn_local_1213_v2_overfit.py
Spatial GNN refiner (kNN graph from spatial coordinates).

train_gnn_local_ccc_1215.py
CCC-enhanced GNN refiner (kNN + ligand–receptor communication features).

SLURM run scripts

finetuen.sbatch
SLURM script to run finetune_ours.py.

train_gnn_local_1213_v2_overfit.slurm
SLURM script to run the spatial GNN refiner.

train_gnn_local_ccc_1215.slurm
SLURM script to run the CCC-enhanced refiner.

Inference scripts

inference_gnn_1114.py (+ .slurm)
Runs inference for base/GNN pipelines (produces predicted vs truth outputs).

inference_gnn_ccc_attn_test.py (+ .slurm)
Runs inference for CCC-enhanced pipeline and tests CCC attention behavior.

Evaluation / utilities

eval.py, evaluate_test_accuracy.py
Compute evaluation metrics (e.g., PCC, MSE, R²) and comparisons across models.

save_test_pred_and_truth_h5ad_1113.py
Saves predictions + ground truth into an .h5ad for analysis/plotting.

utils.py, tokenizer/, configs/
Helpers for tokenization, vocabularies, configs, and shared utilities.

Vocab / assets

rna_vocab_ours_canonical.json, atac_vocab_ours.json, chr_vocab_ours.json, gene2chr_ours_canonical.json, rna_alias2base.json
Vocabularies + mapping files used to match the model’s tokenization.

pretrain_shrunk_for_ours_chr.pth
Pretrained checkpoint used as initialization for fine-tuning.