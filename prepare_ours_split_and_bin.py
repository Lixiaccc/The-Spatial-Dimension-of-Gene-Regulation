#!/usr/bin/env python

import os
import numpy as np
import scanpy as sc
from pathlib import Path

# this import is from the EpiFoundation repo
from data.preprocess import Preprocessor


# =========================
#  paths
# =========================
BASE_DIR = Path(__file__).resolve().parent

raw_dir   = BASE_DIR / "data" / "ours" / "raw"
train_dir = BASE_DIR / "data" / "ours" / "train"
valid_dir = BASE_DIR / "data" / "ours" / "valid"
test_dir  = BASE_DIR / "data" / "ours" / "test"

for d in [train_dir, valid_dir, test_dir]:
    d.mkdir(parents=True, exist_ok=True)

rna_raw_path  = raw_dir / "rna_mouse_human_union_merged_summed_dups.h5ad"
atac_raw_path = raw_dir / "atac_integrated_features.h5ad"

print("RNA raw path :", rna_raw_path)
print("ATAC raw path:", atac_raw_path)

# =========================
#  1. Load and align cells
# =========================
adata_rna  = sc.read_h5ad(rna_raw_path)
adata_atac = sc.read_h5ad(atac_raw_path)

print("RNA shape  (before align):", adata_rna.shape)
print("ATAC shape (before align):", adata_atac.shape)

# intersect by cell barcodes
common_cells = np.intersect1d(adata_rna.obs_names, adata_atac.obs_names)
print("Common cells:", len(common_cells))

adata_rna  = adata_rna[common_cells].copy()
adata_atac = adata_atac[common_cells].copy()

print("RNA shape  (aligned):", adata_rna.shape)
print("ATAC shape (aligned):", adata_atac.shape)

# ensure a 'batch' column exists (needed by Preprocessor)
if "batch" not in adata_rna.obs.columns:
    adata_rna.obs["batch"] = "ours"


# =========================
#  2. Split into train/valid/test
# =========================
def split_adata(adata, ratios=None, seed=0):
    """
    Split AnnData into train / valid / test by cell index.
    ratios is a dict like {"train":0.8, "valid":0.1, "test":0.1}.
    """
    if ratios is None:
        ratios = {"train": 0.8, "valid": 0.1, "test": 0.1}
    assert abs(sum(ratios.values()) - 1.0) < 1e-6

    n = adata.n_obs
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    n_train = int(n * ratios["train"])
    n_valid = int(n * ratios["valid"])
    n_test  = n - n_train - n_valid  # rest

    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train + n_valid]
    test_idx  = idx[n_train + n_valid:]

    ad_train = adata[train_idx].copy()
    ad_valid = adata[valid_idx].copy()
    ad_test  = adata[test_idx].copy()

    return ad_train, ad_valid, ad_test


# split RNA
rna_train, rna_valid, rna_test = split_adata(adata_rna, ratios={"train":0.8, "valid":0.1, "test":0.1}, seed=0)
print("RNA train:", rna_train.shape)
print("RNA valid:", rna_valid.shape)
print("RNA test :", rna_test.shape)

# split ATAC in the same way by matching obs_names
def subset_atac_like_rna(rna_split, atac_full):
    cells = rna_split.obs_names
    return atac_full[cells].copy()

atac_train = subset_atac_like_rna(rna_train, adata_atac)
atac_valid = subset_atac_like_rna(rna_valid, adata_atac)
atac_test  = subset_atac_like_rna(rna_test,  adata_atac)

print("ATAC train:", atac_train.shape)
print("ATAC valid:", atac_valid.shape)
print("ATAC test :", atac_test.shape)

# save raw splits (before binning)
rna_train.write(train_dir / "rna_raw.h5ad")
rna_valid.write(valid_dir / "rna_raw.h5ad")
rna_test.write(test_dir  / "rna_raw.h5ad")

atac_train.write(train_dir / "atac_raw.h5ad")
atac_valid.write(valid_dir / "atac_raw.h5ad")
atac_test.write(test_dir  / "atac_raw.h5ad")

print("Saved raw train/valid/test splits.")


# =========================
#  3. Bin RNA (binRNA step)
# =========================
# This converts continuous RNA counts into small integer bins per gene
# (like categories), which is what EpiFoundation expects.

processor = Preprocessor(
    use_key="X",                  # use main matrix
    filter_gene_by_counts=False,
    filter_cell_by_counts=False,
    normalize_total=False,
    result_normed_key="X_normed",
    log1p=False,
    result_log1p_key="X_log1p",
    subset_hvg=False,
    hvg_use_key=None,
    hvg_flavor="seurat_v3",
    binning=2,                    # 2-bin scheme (same as their example)
    result_binned_key="X_binned"
)

def bin_and_save(rna_split, out_dir, name="ours_rna_binning_2.h5ad"):
    print(f"BinRNA for {out_dir.name} ...")
    processor(rna_split, batch_key="batch")
    out_path = out_dir / name
    rna_split.write(out_path)
    print("  wrote:", out_path)

bin_and_save(rna_train, train_dir)
bin_and_save(rna_valid, valid_dir)
bin_and_save(rna_test,  test_dir)

print("Done: binning finished.")
print("You now have, for each of train/valid/test:")
print("  rna_raw.h5ad           (continuous)")
print("  ours_rna_binning_2.h5ad (binned)")
print("  atac_raw.h5ad")

