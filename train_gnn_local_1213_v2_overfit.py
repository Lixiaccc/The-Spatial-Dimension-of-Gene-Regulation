#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import json
import argparse
import random

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import scanpy as sc
import anndata as ad

import torch
from torch import nn
from torch.utils.data import DataLoader

import yaml
from sklearn.neighbors import NearestNeighbors

from model import EpiFoundation
from tokenizer import GeneVocab
from data.dataloader import PairedSCDataset

# ---- NEW (optional): silence torchtext deprecation warning ----
try:
    import torchtext
    torchtext.disable_torchtext_deprecation_warning()
except Exception:
    pass


# ============================================================
# 0) Utils
# ============================================================

def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])

def masked_flat_metrics(pred_mat, truth_mat, mask_bool):
    a = pred_mat[mask_bool].ravel()
    b = truth_mat[mask_bool].ravel()
    mse = float(np.mean((a - b) ** 2)) if a.size else float("nan")
    pcc = pearson_safe(a, b)
    return mse, pcc

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[INFO] load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}")
    return ckpt


# ============================================================
# 1) Dataset builder (match finetune tokenization distribution)
# ============================================================

def build_paired_dataset(
    rna_file: str,
    atac_file: str,
    rna_key: str,
    atac_key: str,
    config: dict,
    rna_vocab: GeneVocab,
    atac_vocab: GeneVocab,
    cell_vocab: GeneVocab,
    batch_vocab: GeneVocab,
    chr_vocab: GeneVocab,
):
    train_config = config["train"]
    vocab_config = config["vocab"]
    pad = vocab_config["special_tokens"]["pad"]
    cls = vocab_config["special_tokens"]["cls"]

    ds = PairedSCDataset(
        rna_file=rna_file,
        atac_file=atac_file,
        rna_key=rna_key,
        atac_key=atac_key,
        rna_vocab=rna_vocab,
        atac_vocab=atac_vocab,
        cell_vocab=cell_vocab,
        batch_vocab=batch_vocab,
        chr_vocab=chr_vocab,
        gene2chr_file=vocab_config["gene2chr_path"],
        rna_max_len=train_config["model"]["rna_max_len"],
        atac_max_len=train_config["model"]["atac_max_len"],
        pad_token=pad["token"],
        rna_pad_value=pad["value"],
        cls_token=cls["token"],
        logger=None,
        get_full_genes=False,  # match finetune distribution
    )
    return ds


# ============================================================
# 2) Load truth gene matrix + gene-id mapping for HVG3000 order
# ============================================================

def load_truth_gene_matrix(rna_h5ad_path: str, layer_key: str = "X_binned"):
    """
    Returns:
      truth_gene: (N, G) float32
      gene_names: length G (var_names)
    """
    rna = sc.read_h5ad(rna_h5ad_path)
    if layer_key in rna.layers:
        X = rna.layers[layer_key]
    else:
        X = rna.X
    truth = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    truth = truth.astype(np.float32)
    gene_names = rna.var_names.astype(str).tolist()
    return truth, gene_names, rna.obs_names.astype(str).tolist()

def make_gene_id_to_pos(gene_names: list, rna_vocab: GeneVocab):
    """
    Map RNA vocab gene_id -> column position in gene_names order.
    """
    gene_ids = np.array(rna_vocab(gene_names), dtype=np.int64)
    id2pos = -1 * np.ones((len(rna_vocab),), dtype=np.int32)
    for pos, gid in enumerate(gene_ids):
        if 0 <= gid < id2pos.shape[0]:
            id2pos[gid] = pos
    return gene_ids, id2pos


# ============================================================
# 3) EpiFoundation forward pass -> gene-aligned predictions
# ============================================================

@torch.no_grad()
def epifoundation_collect_gene_aligned(
    model,
    loader,
    device,
    rna_vocab,
    atac_vocab,
    pad_token_str,
    cls_token_str,
    id2pos: np.ndarray,
    G: int,
    amp=True,
):
    """
    Returns:
      pred_gene : (N, G) float32  -- base predictions aligned to gene order
      mask_gene : (N, G) bool     -- which genes were actually present in sampled tokens
      cell_emb  : (N, D) float32
    """
    model.eval()

    pred_rows = []
    mask_rows = []
    emb_rows = []

    pad_id = rna_vocab[pad_token_str]
    cls_id = rna_vocab[cls_token_str]

    for bi, batch in enumerate(loader, start=1):
        if bi == 1 or bi % 10 == 0 or bi == len(loader):
            print(f"[INFO] batch {bi}/{len(loader)}")

        rna_ids    = batch["rna_ids"].to(device)      # (B, L)
        atac_ids   = batch["atac_ids"].to(device)
        batch_ids  = batch["batch_ids"].to(device)
        rna_chrs   = batch["rna_chrs"].to(device)
        atac_chrs  = batch["atac_chrs"].to(device)

        padding_positions = atac_ids.eq(atac_vocab[pad_token_str])

        autocast_ctx = torch.cuda.amp.autocast(enabled=amp and device.type == "cuda", dtype=torch.bfloat16)
        with autocast_ctx:
            out = model(
                atac=atac_ids,
                rna=rna_ids,
                src_key_padding_mask=padding_positions,
                batch_id=batch_ids,
                rna_chrs=rna_chrs,
                atac_chrs=atac_chrs,
            )

        value_pred = out["value_pred"]  # (B, L)
        cell_emb = out.get("cell_emb", None)
        if cell_emb is None:
            raise RuntimeError("Model output has no 'cell_emb'. Check model forward() outputs.")

        rna_ids_cpu = rna_ids.detach().cpu().numpy().astype(np.int64)
        pred_cpu = value_pred.detach().cpu().float().numpy().astype(np.float32)

        B, L = rna_ids_cpu.shape
        pred_gene = np.zeros((B, G), dtype=np.float32)
        mask_gene = np.zeros((B, G), dtype=bool)

        for b in range(B):
            gids = rna_ids_cpu[b]
            vals = pred_cpu[b]

            keep = (gids != pad_id) & (gids != cls_id)
            gids = gids[keep]
            vals = vals[keep]

            pos = id2pos[gids]
            ok = pos >= 0
            pos = pos[ok]
            vals = vals[ok]

            pred_gene[b, pos] = vals
            mask_gene[b, pos] = True

        pred_rows.append(pred_gene)
        mask_rows.append(mask_gene)
        emb_rows.append(cell_emb.detach().cpu().float().numpy().astype(np.float32))

    pred_gene_all = np.concatenate(pred_rows, axis=0)
    mask_gene_all = np.concatenate(mask_rows, axis=0)
    emb_all = np.concatenate(emb_rows, axis=0)
    return pred_gene_all, mask_gene_all, emb_all


# ============================================================
# 4) Coords/obs from ATAC *_xy
# ============================================================

def load_coords_and_obs_from_h5ad(h5ad_path: str, xcol: str, ycol: str, split_name: str, n_expected: int):
    adata = sc.read_h5ad(h5ad_path)
    if xcol not in adata.obs.columns or ycol not in adata.obs.columns:
        raise KeyError(f"{split_name}: missing coord columns '{xcol}', '{ycol}' in {h5ad_path}")
    coords = adata.obs[[xcol, ycol]].to_numpy().astype(np.float32)
    if coords.shape[0] != n_expected:
        raise ValueError(f"{split_name}: coords n={coords.shape[0]} != expected n={n_expected}")
    return coords, adata.obs.copy()


# ============================================================
# 5) Graph building
# ============================================================

def build_knn_graph(coords: np.ndarray, k: int, r_max: float = None):
    if coords.shape[0] == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    k_eff = min(k + 1, coords.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(coords)
    dists, idxs = nbrs.kneighbors(coords)

    if k_eff < 2:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    dists = dists[:, 1:]
    idxs  = idxs[:, 1:]
    k_used = dists.shape[1]

    src = np.repeat(np.arange(coords.shape[0]), k_used)
    dst = idxs.reshape(-1)
    edge_dist = dists.reshape(-1)

    if r_max is not None:
        keep = edge_dist <= float(r_max)
        src = src[keep]
        dst = dst[keep]
        edge_dist = edge_dist[keep]

    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    return edge_index, edge_dist.astype(np.float32)

def compute_degrees(edge_index_t: torch.Tensor, n_nodes: int, device):
    src = edge_index_t[0]
    deg = torch.zeros((n_nodes,), device=device, dtype=torch.float32)
    if src.numel() > 0:
        deg = deg.index_add(0, src, torch.ones_like(src, dtype=torch.float32))
    return deg

# ---- NEW: edge dropout to reduce graph memorization ----
def apply_edge_dropout(edge_index_t: torch.Tensor, edge_dist_t: torch.Tensor, drop_p: float, training: bool):
    if (not training) or drop_p <= 0.0:
        return edge_index_t, edge_dist_t
    E = edge_index_t.shape[1]
    if E == 0:
        return edge_index_t, edge_dist_t
    keep = torch.rand((E,), device=edge_index_t.device) > float(drop_p)
    if keep.sum() == 0:
        keep[torch.randint(0, E, (1,), device=edge_index_t.device)] = True
    return edge_index_t[:, keep], edge_dist_t[keep]


# ============================================================
# 6) GNN refiner (gene-level)
#    (ONLY change here: add dropout)
# ============================================================

class ResidualGNNLayer(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_dist, degrees):
        src, dst = edge_index
        N, D = x.shape
        if src.numel() == 0:
            return x

        msg = x[dst]
        agg = torch.zeros((N, D), device=x.device, dtype=x.dtype)
        agg = agg.index_add(0, src, msg)

        deg = degrees.clamp_min(1.0).unsqueeze(-1)
        agg = agg / deg

        dist_sum = torch.zeros((N,), device=x.device, dtype=x.dtype)
        dist_sum = dist_sum.index_add(0, src, edge_dist)
        mean_dist = (dist_sum / degrees.clamp_min(1.0)).unsqueeze(-1)

        feat = torch.cat([x, agg, degrees.unsqueeze(-1), mean_dist], dim=-1)
        out = self.mlp(feat)
        out = self.norm(x + out)
        return out

class ResidualGNN(nn.Module):
    def __init__(self, dim, hidden_dim, n_layers=2, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([ResidualGNNLayer(dim, hidden_dim, dropout=dropout) for _ in range(n_layers)])

    def forward(self, x, edge_index, edge_dist, degrees):
        for layer in self.layers:
            x = layer(x, edge_index, edge_dist, degrees)
        return x

class GenePredEncoder(nn.Module):
    """Project base gene-vector (G) -> P for node features."""
    def __init__(self, G, out_dim=256, hidden=512, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(G, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, y):
        return self.net(y)

class GeneDeltaHead(nn.Module):
    """Predict delta gene-vector (G)."""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 7) Plotting
# ============================================================

def save_curves(out_png_loss: str, out_png_pcc: str, history: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    tr_loss = history["train_loss"]
    va_loss = history["val_loss"]
    va_pcc  = history["val_pcc"]

    plt.figure()
    plt.plot(epochs, tr_loss, label="train_loss")
    plt.plot(epochs, va_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("masked MSE loss")
    plt.title("GNN gene refiner loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_loss, dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, va_pcc, label="val_masked_PCC")
    plt.xlabel("epoch")
    plt.ylabel("PCC")
    plt.title("GNN gene refiner validation PCC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_pcc, dpi=200)
    plt.close()


# ============================================================
# 8) Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)

    ap.add_argument("--rna_train", required=True)
    ap.add_argument("--atac_train", required=True)
    ap.add_argument("--rna_val", required=True)
    ap.add_argument("--atac_val", required=True)

    ap.add_argument("--rna_key", required=True, help="e.g. X_binned")
    ap.add_argument("--atac_key", required=True, help="e.g. X")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--x_col", type=str, default="x")
    ap.add_argument("--y_col", type=str, default="y")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--r_max", type=float, default=None)

    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--gnn_hidden", type=int, default=256)
    ap.add_argument("--gnn_layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--early_stop_patience", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=10)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)

    ap.add_argument("--save_h5ad", required=True)
    ap.add_argument("--save_ckpt", default="best_gene_gnn_refiner.pth")
    ap.add_argument("--out_dir", default="gnn_gene_refiner_out")

    ap.add_argument("--clip_min", type=float, default=0.0)
    ap.add_argument("--clip_max", type=float, default=None)

    # ---- NEW REGULARIZATION ARGS (added, nothing removed) ----
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--edge_dropout", type=float, default=0.2)
    ap.add_argument("--delta_l2", type=float, default=1e-3)

    # optional LR scheduler
    ap.add_argument("--lr_patience", type=int, default=20)
    ap.add_argument("--lr_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)

    config = load_yaml(args.config)
    train_config = config["train"]
    data_config = config["data"]
    vocab_config = config["vocab"]

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    print("[INFO] Loading vocabularies...")
    rna_vocab = GeneVocab.from_file(vocab_config["rna_path"])
    atac_vocab = GeneVocab.from_file(vocab_config["atac_path"])
    cell_vocab = GeneVocab.from_file(vocab_config["cell_type_path"])
    batch_vocab = GeneVocab.from_file(vocab_config["batch_path"])
    chr_vocab = GeneVocab.from_file(vocab_config["chr_path"])

    pad = vocab_config["special_tokens"]["pad"]
    cls = vocab_config["special_tokens"]["cls"]

    print(f"[INFO] RNA vocab size:  {len(rna_vocab)}")
    print(f"[INFO] ATAC vocab size: {len(atac_vocab)}")
    print(f"[INFO] CHR vocab size:  {len(chr_vocab)}")

    print("\n" + "="*70)
    print("STEP 0: LOAD GENE-SPACE TRUTH (RNA)")
    print("="*70)
    tr_truth_gene, gene_names, tr_cells = load_truth_gene_matrix(args.rna_train, layer_key=args.rna_key)
    va_truth_gene, gene_names2, va_cells = load_truth_gene_matrix(args.rna_val, layer_key=args.rna_key)
    assert gene_names == gene_names2, "Train/Val gene order mismatch in RNA var_names."
    G = len(gene_names)
    print(f"[INFO] Gene dim G={G}")
    print(f"[INFO] Train truth shape: {tr_truth_gene.shape}, Val truth shape: {va_truth_gene.shape}")

    _, id2pos = make_gene_id_to_pos(gene_names, rna_vocab)

    print("\n" + "="*70)
    print("STEP A: BUILD PAIRED DATASETS (EpiFoundation inputs)")
    print("="*70)
    ds_tr = build_paired_dataset(
        rna_file=args.rna_train, atac_file=args.atac_train,
        rna_key=args.rna_key, atac_key=args.atac_key,
        config=config, rna_vocab=rna_vocab, atac_vocab=atac_vocab,
        cell_vocab=cell_vocab, batch_vocab=batch_vocab, chr_vocab=chr_vocab
    )
    ds_va = build_paired_dataset(
        rna_file=args.rna_val, atac_file=args.atac_val,
        rna_key=args.rna_key, atac_key=args.atac_key,
        config=config, rna_vocab=rna_vocab, atac_vocab=atac_vocab,
        cell_vocab=cell_vocab, batch_vocab=batch_vocab, chr_vocab=chr_vocab
    )
    gc.collect()

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("\n" + "="*70)
    print("STEP B: LOAD EPIFOUNDATION")
    print("="*70)

    # IMPORTANT: SAME init as your original (to match checkpoint)
    model = EpiFoundation(
        num_class_cell=len(cell_vocab),
        num_rnas=len(rna_vocab),
        num_atacs=len(atac_vocab),
        num_values=data_config["bin_num"],
        num_chrs=len(chr_vocab),
        embed_dim=train_config["model"]["embedding_dim"],
        depth=train_config["model"]["num_layers"],
        heads=train_config["model"]["head_num"],
        head_dim=train_config["model"]["head_dim"],
        encoder=train_config["model"]["encoder"],
        dropout=train_config["model"]["dropout"],
        pad_token_idx_rna=rna_vocab[pad["token"]],
        pad_token_idx_atac=atac_vocab[pad["token"]],
        cell_emb_style=train_config["model"]["cell_emb_style"],
        mvc_arch_style=train_config["model"]["mvc_arch_style"],
        use_batch_labels=train_config["model"]["use_batch_labels"],
        batch_label_num=len(batch_vocab),
        use_chr_labels=train_config["model"]["use_chr_labels"],
        stage="value_finetune",
    ).to(device)

    _ = load_checkpoint_into_model(model, args.checkpoint, device)
    model.eval()
    print("[INFO] EpiFoundation loaded.")

    print("\n" + "="*70)
    print("STEP C: EPIFOUNDATION BASELINE (GENE-ALIGNED)")
    print("="*70)

    tr_pred_gene, tr_mask_gene, tr_emb = epifoundation_collect_gene_aligned(
        model, dl_tr, device, rna_vocab, atac_vocab,
        pad["token"], cls["token"], id2pos=id2pos, G=G, amp=True
    )
    va_pred_gene, va_mask_gene, va_emb = epifoundation_collect_gene_aligned(
        model, dl_va, device, rna_vocab, atac_vocab,
        pad["token"], cls["token"], id2pos=id2pos, G=G, amp=True
    )

    tr_mse_base, tr_pcc_base = masked_flat_metrics(tr_pred_gene, tr_truth_gene, tr_mask_gene)
    va_mse_base, va_pcc_base = masked_flat_metrics(va_pred_gene, va_truth_gene, va_mask_gene)

    print(f"[BASE][TRAIN] gene-space masked MSE: {tr_mse_base:.4f}")
    print(f"[BASE][TRAIN] gene-space masked PCC: {tr_pcc_base:.4f}")
    print(f"[BASE][VAL]   gene-space masked MSE: {va_mse_base:.4f}")
    print(f"[BASE][VAL]   gene-space masked PCC: {va_pcc_base:.4f}")

    Ntr = tr_pred_gene.shape[0]
    Nva = va_pred_gene.shape[0]

    tr_coords, tr_obs = load_coords_and_obs_from_h5ad(args.atac_train, args.x_col, args.y_col, "TRAIN", Ntr)
    va_coords, va_obs = load_coords_and_obs_from_h5ad(args.atac_val, args.x_col, args.y_col, "VAL", Nva)

    print("\n" + "="*70)
    print("STEP D: BUILD kNN GRAPHS")
    print("="*70)
    tr_edge_index_np, tr_edge_dist_np = build_knn_graph(tr_coords, k=args.k, r_max=args.r_max)
    va_edge_index_np, va_edge_dist_np = build_knn_graph(va_coords, k=args.k, r_max=args.r_max)
    print(f"[INFO] TRAIN edges: {tr_edge_index_np.shape[1]:,}")
    print(f"[INFO] VAL   edges: {va_edge_index_np.shape[1]:,}")

    tr_emb_t = torch.from_numpy(tr_emb).to(device)
    va_emb_t = torch.from_numpy(va_emb).to(device)

    tr_base_t = torch.from_numpy(tr_pred_gene).to(device)
    va_base_t = torch.from_numpy(va_pred_gene).to(device)

    tr_y_t = torch.from_numpy(tr_truth_gene).to(device)
    va_y_t = torch.from_numpy(va_truth_gene).to(device)

    tr_mask_t = torch.from_numpy(tr_mask_gene).to(device)
    va_mask_t = torch.from_numpy(va_mask_gene).to(device)

    tr_ei_t = torch.from_numpy(tr_edge_index_np).long().to(device)
    tr_ed_t = torch.from_numpy(tr_edge_dist_np).float().to(device)
    va_ei_t = torch.from_numpy(va_edge_index_np).long().to(device)
    va_ed_t = torch.from_numpy(va_edge_dist_np).float().to(device)

    tr_deg = compute_degrees(tr_ei_t, Ntr, device)
    va_deg = compute_degrees(va_ei_t, Nva, device)

    print("\n" + "="*70)
    print("STEP E: TRAIN GNN GENE REFINER (REGULARIZED)")
    print("="*70)

    D = tr_emb.shape[1]
    P = int(args.proj_dim)
    alpha = float(args.alpha)

    # only change: add dropout into these modules
    gene_enc = GenePredEncoder(G=G, out_dim=P, hidden=max(512, P*2), dropout=args.dropout).to(device)
    gnn_in_dim = D + P
    gnn = ResidualGNN(dim=gnn_in_dim, hidden_dim=args.gnn_hidden, n_layers=args.gnn_layers, dropout=args.dropout).to(device)
    head = GeneDeltaHead(in_dim=gnn_in_dim, hidden_dim=args.gnn_hidden, out_dim=G, dropout=args.dropout).to(device)

    params = list(gene_enc.parameters()) + list(gnn.parameters()) + list(head.parameters())

    # NEW: AdamW + weight decay
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # NEW: scheduler (optional)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr, verbose=True
    )

    def masked_mse_loss(pred, truth, mask_bool):
        diff = pred - truth
        diff = diff[mask_bool]
        return torch.mean(diff * diff)

    def refine(emb_t, base_gene_t, ei_t, ed_t, deg_t, training=False, return_delta=False):
        # NEW: edge dropout during training
        ei_use, ed_use = apply_edge_dropout(ei_t, ed_t, args.edge_dropout, training=training)

        z = gene_enc(base_gene_t)          # (N, P)
        x = torch.cat([emb_t, z], dim=1)   # (N, D+P)
        x2 = gnn(x, ei_use, ed_use, deg_t)
        delta = head(x2)                   # (N, G)

        has_nb = (deg_t > 0).float().unsqueeze(-1)
        delta = delta * has_nb

        y = base_gene_t + alpha * delta

        if args.clip_min is not None:
            y = torch.clamp(y, min=float(args.clip_min))
        if args.clip_max is not None:
            y = torch.clamp(y, max=float(args.clip_max))

        if return_delta:
            return y, delta
        return y

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_pcc": [], "lr": []}
    best_val_pcc = -1e9
    best_epoch = -1
    patience = 0

    print("Epoch | TrainLoss | ValLoss | ValPCC(masked) | BaseValPCC | LR")
    print("-"*90)
    print(f"[BASE][VAL] masked PCC: {va_pcc_base:.4f}")

    for epoch in range(1, args.epochs + 1):
        gene_enc.train(); gnn.train(); head.train()
        opt.zero_grad()

        pred_tr, delta_tr = refine(tr_emb_t, tr_base_t, tr_ei_t, tr_ed_t, tr_deg, training=True, return_delta=True)

        loss_mse = masked_mse_loss(pred_tr, tr_y_t, tr_mask_t)
        # NEW: delta penalty (keeps corrections small)
        loss_delta = torch.mean(delta_tr * delta_tr)
        loss_tr = loss_mse + float(args.delta_l2) * loss_delta

        loss_tr.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        gene_enc.eval(); gnn.eval(); head.eval()
        with torch.no_grad():
            pred_va = refine(va_emb_t, va_base_t, va_ei_t, va_ed_t, va_deg, training=False)
            loss_va = masked_mse_loss(pred_va, va_y_t, va_mask_t).item()

            pred_va_np = pred_va.detach().cpu().numpy()
            _, pcc_va = masked_flat_metrics(pred_va_np, va_truth_gene, va_mask_gene)

        # NEW: step scheduler on val loss
        scheduler.step(loss_va)

        cur_lr = float(opt.param_groups[0]["lr"])
        history["epoch"].append(epoch)
        history["train_loss"].append(float(loss_tr.item()))
        history["val_loss"].append(float(loss_va))
        history["val_pcc"].append(float(pcc_va))
        history["lr"].append(cur_lr)

        if epoch == 1 or epoch % args.log_every == 0:
            print(f"{epoch:5d} | {loss_tr.item():9.4f} | {loss_va:7.4f} | {pcc_va:14.4f} | {va_pcc_base:10.4f} | {cur_lr:.2e}")

        if pcc_va > best_val_pcc + 1e-4:
            best_val_pcc = pcc_va
            best_epoch = epoch
            patience = 0
            torch.save({
                "epoch": epoch,
                "alpha": alpha,
                "proj_dim": P,
                "G": G,
                "gene_enc": gene_enc.state_dict(),
                "gnn": gnn.state_dict(),
                "head": head.state_dict(),
                "config": vars(args),
                "best_val_pcc": float(best_val_pcc),
                "gene_names": gene_names,
            }, args.save_ckpt)
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"[EARLY STOP] no improvement for {args.early_stop_patience} epochs. best_epoch={best_epoch}, best_val_pcc={best_val_pcc:.4f}")
                break

    loss_png = os.path.join(args.out_dir, "loss_curve.png")
    pcc_png  = os.path.join(args.out_dir, "pcc_curve.png")
    hist_json = os.path.join(args.out_dir, "history.json")
    with open(hist_json, "w") as f:
        json.dump(history, f, indent=2)
    save_curves(loss_png, pcc_png, history)
    print(f"[INFO] Saved curves: {loss_png}, {pcc_png}")
    print(f"[INFO] Saved history: {hist_json}")

    print("\n" + "="*70)
    print("STEP F: FINAL EVAL + SAVE h5ad (VAL)")
    print("="*70)

    if os.path.exists(args.save_ckpt):
        best = torch.load(args.save_ckpt, map_location=device)
        gene_enc.load_state_dict(best["gene_enc"])
        gnn.load_state_dict(best["gnn"])
        head.load_state_dict(best["head"])
        print(f"[INFO] Loaded best refiner: epoch={best['epoch']} best_val_pcc={best.get('best_val_pcc', best_val_pcc):.4f}")
    else:
        print("[WARN] No refiner checkpoint saved; using last weights.")

    gene_enc.eval(); gnn.eval(); head.eval()
    with torch.no_grad():
        pred_va_best = refine(va_emb_t, va_base_t, va_ei_t, va_ed_t, va_deg, training=False).detach().cpu().numpy()
        pred_tr_best = refine(tr_emb_t, tr_base_t, tr_ei_t, tr_ed_t, tr_deg, training=False).detach().cpu().numpy()

    tr_mse_ref, tr_pcc_ref = masked_flat_metrics(pred_tr_best, tr_truth_gene, tr_mask_gene)
    va_mse_ref, va_pcc_ref = masked_flat_metrics(pred_va_best, va_truth_gene, va_mask_gene)

    print(f"[REFINED][TRAIN] masked PCC: {tr_pcc_ref:.4f} (base {tr_pcc_base:.4f})")
    print(f"[REFINED][VAL]   masked PCC: {va_pcc_ref:.4f} (base {va_pcc_base:.4f})")

    var = pd.DataFrame(index=gene_names)
    obs = va_obs.copy()

    X = np.zeros((Nva, G), dtype=np.float32)
    out = ad.AnnData(X=X, obs=obs, var=var)

    out.layers["pred_base"] = va_pred_gene.astype(np.float32)
    out.layers["pred_refine"] = pred_va_best.astype(np.float32)
    out.layers["truth_value"] = va_truth_gene.astype(np.float32)
    out.layers["mask_predicted_genes"] = va_mask_gene.astype(np.uint8)

    out.obsm["X_cell_emb"] = va_emb.astype(np.float32)

    out.uns["eval_info"] = {
        "space": "gene",
        "gene_order": "rna.var_names from RNA h5ad",
        "base_val_masked_pcc": float(va_pcc_base),
        "refined_val_masked_pcc": float(va_pcc_ref),
        "base_val_masked_mse": float(va_mse_base),
        "refined_val_masked_mse": float(va_mse_ref),
        "k": int(args.k),
        "r_max": args.r_max,
        "gnn_node_features": "concat(cell_emb, proj(base_gene_vector))",
        "alpha": float(alpha),
        "proj_dim": int(P),
        "best_epoch": int(best_epoch),
        "clip_min": args.clip_min,
        "clip_max": args.clip_max,
        "regularization": {
            "optimizer": "AdamW",
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "edge_dropout": float(args.edge_dropout),
            "delta_l2": float(args.delta_l2),
            "lr_scheduler": "ReduceLROnPlateau",
            "lr_patience": int(args.lr_patience),
            "lr_factor": float(args.lr_factor),
            "min_lr": float(args.min_lr),
        }
    }

    os.makedirs(os.path.dirname(args.save_h5ad) or ".", exist_ok=True)
    print(f"[INFO] Saving h5ad: {args.save_h5ad}")
    out.write_h5ad(args.save_h5ad)
    print("[DONE]")


if __name__ == "__main__":
    main()

