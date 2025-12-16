# EpiFoundation Spatial Refinement  
### ATAC → RNA Gene Expression Prediction with GNNs and Cell–Cell Communication

This repository contains the code for our course/research project:

> **“The Spatial Dimension of Gene Regulation: Integrating ATAC-seq with Tissue Context Using a Graph Neural Network for Gene Expression Predictions.”**

We study how spatial tissue context and cell–cell communication can improve gene expression prediction from chromatin accessibility by combining **EpiFoundation** with **graph neural networks (GNNs)**.

---

## Project Overview

We predict **gene expression (RNA)** from **chromatin accessibility (ATAC)** using a **two-stage modeling pipeline**:

### 1. Base Model — *EpiFoundation Fine-Tuning*
- Fine-tunes a pretrained EpiFoundation model on paired **spatial ATAC-RNA** data
- Produces:
  - Per-cell gene expression predictions
  - Latent cell embeddings

### 2. Spatial Refinement — *Graph Neural Networks*
The base predictions are refined using spatial neighborhood information via one of the following approaches:

- **Spatial kNN GNN**  
  Uses a k-nearest-neighbor graph constructed from spatial coordinates

- **CCC-Enhanced GNN**  
  Extends the spatial GNN with **cell–cell communication (CCC)** features derived from **LIANA** ligand–receptor inference

---

## Repository Structure

### Core Training Scripts

- **`finetune_ours.py`**  
  Fine-tunes EpiFoundation on paired spatial ATAC-RNA data (base model).

- **`train_gnn_local_1213_v2_overfit.py`**  
  Spatial GNN refiner using a kNN graph constructed from spatial coordinates.

- **`train_gnn_local_ccc_1215.py`**  
  CCC-enhanced GNN refiner incorporating ligand–receptor communication features.

---

### SLURM Job Scripts

Scripts for running training and inference on HPC clusters:

- **`finetune.sbatch`**  
  Runs `finetune_ours.py`.

- **`train_gnn_local_1213_v2_overfit.slurm`**  
  Runs the spatial GNN refiner.

- **`train_gnn_local_ccc_1215.slurm`**  
  Runs the CCC-enhanced GNN refiner.

---

### Inference Scripts

- **`inference_gnn_1114.py`** (+ `.slurm`)  
  Runs inference for the base and spatial GNN pipelines, producing predicted vs. ground-truth outputs.

- **`inference_gnn_ccc_attn_test.py`** (+ `.slurm`)  
  Runs inference for the CCC-enhanced pipeline and evaluates CCC attention behavior.

---

### Evaluation & Utilities

- **`eval.py`, `evaluate_test_accuracy.py`**  
  Compute evaluation metrics such as:
  - Pearson correlation (PCC)
  - Mean squared error (MSE)
  - R²

- **`save_test_pred_and_truth_h5ad_1113.py`**  
  Saves predictions and ground truth into an `.h5ad` file for downstream analysis and visualization.

- **`utils.py`, `tokenizer/`, `configs/`**  
  Shared utilities for tokenization, vocab handling, configuration management, and helper functions.

---

### Vocabularies & Model Assets

- **Vocabulary & mapping files**
  - `rna_vocab_ours_canonical.json`
  - `atac_vocab_ours.json`
  - `chr_vocab_ours.json`
  - `gene2chr_ours_canonical.json`
  - `rna_alias2base.json`

- **Pretrained checkpoint**
  - `pretrain_shrunk_for_ours_chr.pth`  
    Used as initialization for EpiFoundation fine-tuning.

---

## Data Availability

All processed datasets and required inputs are available here:

🔗 **Google Drive:**  
https://drive.google.com/drive/folders/1xsziV5dKn1OpicRk0XZyA-heWbwc2NvT

---

## Notes

- This repository is research-oriented and optimized for HPC/SLURM workflows.
- Scripts assume familiarity with EpiFoundation, AnnData (`.h5ad`), and graph-based deep learning.
- File naming reflects active experimentation and ablation studies during development.

---

## Citation

If you use or build upon this work, please cite:

> *The Spatial Dimension of Gene Regulation: Integrating ATAC-seq with Tissue Context Using a Graph Neural Network for Gene Expression Predictions.*

---

