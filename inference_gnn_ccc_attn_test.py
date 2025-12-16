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

# ---- OPTIONAL: silence torchtext deprecation warning ----
try:
    import torchtext
    torchtext.disable_torchtext_deprecation_warning()
except Exception:
    pass


# ============================================================
# Utils
# ============================================================

def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

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

def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    print(f"[INFO] Loading EpiFoundation checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    new_state = {}
    for k, v in state.items():
        new_state[k[len("module."):] if k.startswith("module.") else k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[INFO] load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}")
    return ckpt


# ============================================================
# Dataset builder
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
        get_full_genes=False,   # IMPORTANT: match finetune distribution
    )
    return ds


# ============================================================
# Truth gene matrix in HVG order (G=3000)
# ============================================================

def load_truth_gene_matrix(rna_h5ad_path: str, layer_key: str = "X_binned"):
    rna = sc.read_h5ad(rna_h5ad_path)
    X = rna.layers[layer_key] if layer_key in rna.layers else rna.X
    truth = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    truth = truth.astype(np.float32)
    gene_names = rna.var_names.astype(str).tolist()
    obs = rna.obs.copy()
    obs_names = rna.obs_names.astype(str).tolist()
    return truth, gene_names, obs, obs_names

def apply_alias2base_to_gene_names(gene_names: list):
    alias2base_path = "/insomnia001/depts/msph/users/lc3716/EpiFoundation/rna_alias2base.json"
    if os.path.exists(alias2base_path):
        with open(alias2base_path, "r") as f:
            alias2base = json.load(f)
        mapped = [alias2base.get(g, g) for g in gene_names]
        print(f"[INFO] Applied alias→base mapping to RNA var_names ({len(alias2base)} aliases).")
        return mapped
    return gene_names

def make_gene_id_to_pos(gene_names: list, rna_vocab: GeneVocab):
    gene_ids = np.array(rna_vocab(gene_names), dtype=np.int64)
    id2pos = -1 * np.ones((len(rna_vocab),), dtype=np.int32)
    for pos, gid in enumerate(gene_ids):
        if 0 <= gid < id2pos.shape[0]:
            id2pos[gid] = pos
    return gene_ids, id2pos


# ============================================================
# EpiFoundation forward -> gene-aligned (G=3000)
# ============================================================

@torch.no_grad()
def epifoundation_collect_gene_aligned_from_paired(
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
    model.eval()

    pad_id = rna_vocab[pad_token_str]
    cls_id = rna_vocab[cls_token_str]

    pred_rows = []
    mask_rows = []
    emb_rows = []

    use_amp = (device.type == "cuda") and bool(amp)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    for bi, batch in enumerate(loader, start=1):
        if bi == 1 or bi % 10 == 0 or bi == len(loader):
            print(f"[INFO] batch {bi}/{len(loader)}")

        rna_ids    = batch["rna_ids"].to(device)
        atac_ids   = batch["atac_ids"].to(device)
        batch_ids  = batch["batch_ids"].to(device)
        rna_chrs   = batch["rna_chrs"].to(device)
        atac_chrs  = batch["atac_chrs"].to(device)

        padding_positions = atac_ids.eq(atac_vocab[pad_token_str])

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            out = model(
                atac=atac_ids,
                rna=rna_ids,
                src_key_padding_mask=padding_positions,
                batch_id=batch_ids,
                rna_chrs=rna_chrs,
                atac_chrs=atac_chrs,
            )

        value_pred = out["value_pred"]   # (B,L)
        cell_emb   = out["cell_emb"]     # (B,D)

        rna_ids_cpu = rna_ids.detach().cpu().numpy().astype(np.int64)
        pred_cpu    = value_pred.detach().cpu().float().numpy().astype(np.float32)

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
# Graph utilities
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
        src, dst, edge_dist = src[keep], dst[keep], edge_dist[keep]

    return np.stack([src, dst], axis=0).astype(np.int64), edge_dist.astype(np.float32)

def compute_degrees(edge_index_t: torch.Tensor, n_nodes: int, device):
    src = edge_index_t[0]
    deg = torch.zeros((n_nodes,), device=device, dtype=torch.float32)
    if src.numel():
        deg = deg.index_add(0, src, torch.ones_like(src, dtype=torch.float32))
    return deg


# ============================================================
# CCC parsing + edge feature building (must match training)
# ============================================================

def _safe_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)

def parse_ccc_out_map(obs: pd.DataFrame):
    """
    out_map[i] = set(j) that i sends to, using obs['sigccc_out_pair_keys'].
    Requires obs['pair_key'] (species-prefixed unique key).
    """
    if "pair_key" not in obs.columns:
        raise KeyError("Missing obs['pair_key'] required for CCC mapping.")

    pair_keys = obs["pair_key"].astype(str).to_numpy()
    pk_to_idx = {pk: i for i, pk in enumerate(pair_keys)}

    out_map = [set() for _ in range(len(obs))]

    if "sigccc_out_pair_keys" not in obs.columns:
        print("[WARN] No sigccc_out_pair_keys in obs; CCC features will be zeros.")
        return out_map, pk_to_idx

    col = obs["sigccc_out_pair_keys"]
    for i in range(len(obs)):
        out_str = _safe_str(col.iloc[i])
        if out_str == "" or out_str.lower() == "nan":
            continue
        for tgt in out_str.split(";"):
            if tgt in pk_to_idx:
                out_map[i].add(pk_to_idx[tgt])

    return out_map, pk_to_idx

def build_ccc_edge_feat_for_knn(edge_index_np: np.ndarray, out_map: list):
    """
    For each directed kNN edge (i -> j):
      ccc_out_ij = 1 if i sends to j
      ccc_in_ij  = 1 if j sends to i  (reverse out)
    Returns (E,2) float32.
    """
    src = edge_index_np[0]
    dst = edge_index_np[1]
    E = src.shape[0]

    ccc_out = np.zeros((E,), dtype=np.float32)
    ccc_in  = np.zeros((E,), dtype=np.float32)

    for e in range(E):
        i = int(src[e]); j = int(dst[e])
        if j in out_map[i]:
            ccc_out[e] = 1.0
        if i in out_map[j]:
            ccc_in[e] = 1.0

    return np.stack([ccc_out, ccc_in], axis=1).astype(np.float32)


# ============================================================
# CCC-attn GNN modules (MUST MATCH TRAINING)
# ============================================================

class EdgeAttention(nn.Module):
    """
    edge_feat = [dist, ccc_out, ccc_in] -> weight in [0,1]
    """
    def __init__(self, edge_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, edge_feat):
        return self.net(edge_feat).squeeze(-1)  # (E,)

class ResidualGNNLayer(nn.Module):
    def __init__(self, dim, hidden_dim, edge_dim=3, dropout=0.0):
        super().__init__()
        self.edge_attn = EdgeAttention(edge_dim=edge_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_feat, degrees):
        src, dst = edge_index
        N, D = x.shape
        if src.numel() == 0:
            return x

        w = self.edge_attn(edge_feat).unsqueeze(-1)  # (E,1)
        msg = x[dst] * w

        agg = torch.zeros((N, D), device=x.device, dtype=x.dtype)
        agg.index_add_(0, src, msg)

        deg = degrees.clamp_min(1.0).unsqueeze(-1)
        agg = agg / deg

        feat = torch.cat([x, agg, degrees.unsqueeze(-1)], dim=-1)
        out = self.mlp(feat)
        return self.norm(x + out)

class ResidualGNN(nn.Module):
    def __init__(self, dim, hidden_dim, n_layers=2, dropout=0.0, edge_dim=3):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualGNNLayer(dim, hidden_dim, edge_dim=edge_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, edge_index, edge_feat, degrees):
        for layer in self.layers:
            x = layer(x, edge_index, edge_feat, degrees)
        return x

class GenePredEncoder(nn.Module):
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
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True)
    ap.add_argument("--epif_ckpt", required=True)
    ap.add_argument("--gnn_ckpt", required=True)

    ap.add_argument("--atac_test", required=True)
    ap.add_argument("--rna_test", required=True)

    ap.add_argument("--rna_key", required=True)
    ap.add_argument("--atac_key", required=True)

    ap.add_argument("--x_col", default="x")
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--r_max", type=float, default=None)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_h5ad", required=True)

    args = ap.parse_args()
    seed_all(args.seed)

    config = load_yaml(args.config)
    train_config = config["train"]
    data_config = config["data"]
    vocab_config = config["vocab"]

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # vocabs
    rna_vocab = GeneVocab.from_file(vocab_config["rna_path"])
    atac_vocab = GeneVocab.from_file(vocab_config["atac_path"])
    cell_vocab = GeneVocab.from_file(vocab_config["cell_type_path"])
    batch_vocab = GeneVocab.from_file(vocab_config["batch_path"])
    chr_vocab = GeneVocab.from_file(vocab_config["chr_path"])

    pad = vocab_config["special_tokens"]["pad"]["token"]
    cls = vocab_config["special_tokens"]["cls"]["token"]

    print("\n" + "="*70)
    print("STEP 1: Load RNA truth (gene-space HVG3000)")
    print("="*70)
    truth_gene, gene_names, _, _ = load_truth_gene_matrix(args.rna_test, layer_key=args.rna_key)
    gene_names = apply_alias2base_to_gene_names(gene_names)
    N, G = truth_gene.shape
    print(f"[INFO] Test truth shape: {truth_gene.shape} (N={N}, G={G})")

    _, id2pos = make_gene_id_to_pos(gene_names, rna_vocab)

    print("\n" + "="*70)
    print("STEP 2: Build PairedSCDataset for test")
    print("="*70)
    ds = build_paired_dataset(
        rna_file=args.rna_test,
        atac_file=args.atac_test,
        rna_key=args.rna_key,
        atac_key=args.atac_key,
        config=config,
        rna_vocab=rna_vocab,
        atac_vocab=atac_vocab,
        cell_vocab=cell_vocab,
        batch_vocab=batch_vocab,
        chr_vocab=chr_vocab,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("\n" + "="*70)
    print("STEP 3: Load ATAC obs/coords (and CCC fields)")
    print("="*70)
    atac = sc.read_h5ad(args.atac_test)
    if atac.n_obs != N:
        raise ValueError(f"ATAC test n_obs={atac.n_obs} != RNA test n_obs={N}. Must match.")
    if args.x_col not in atac.obs.columns or args.y_col not in atac.obs.columns:
        raise KeyError(f"ATAC test missing coords {args.x_col}/{args.y_col}.")
    coords = atac.obs[[args.x_col, args.y_col]].to_numpy().astype(np.float32)
    obs = atac.obs.copy()
    del atac
    gc.collect()

    print("\n" + "="*70)
    print("STEP 4: Run EpiFoundation baseline (token -> gene)")
    print("="*70)
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
        pad_token_idx_rna=rna_vocab[pad],
        pad_token_idx_atac=atac_vocab[pad],
        cell_emb_style=train_config["model"]["cell_emb_style"],
        mvc_arch_style=train_config["model"]["mvc_arch_style"],
        use_batch_labels=train_config["model"]["use_batch_labels"],
        batch_label_num=len(batch_vocab),
        use_chr_labels=train_config["model"]["use_chr_labels"],
        stage="value_finetune",
    ).to(device)
    _ = load_checkpoint_into_model(model, args.epif_ckpt, device)
    model.eval()

    pred_base_gene, mask_gene, cell_emb = epifoundation_collect_gene_aligned_from_paired(
        model=model,
        loader=dl,
        device=device,
        rna_vocab=rna_vocab,
        atac_vocab=atac_vocab,
        pad_token_str=pad,
        cls_token_str=cls,
        id2pos=id2pos,
        G=G,
        amp=True,
    )

    mse_base, pcc_base = masked_flat_metrics(pred_base_gene, truth_gene, mask_gene)
    print(f"[BASE][TEST] gene-space masked MSE: {mse_base:.4f}")
    print(f"[BASE][TEST] gene-space masked PCC: {pcc_base:.4f}")

    print("\n" + "="*70)
    print("STEP 5: Build kNN graph + CCC edge features (dist + out + in)")
    print("="*70)
    edge_index_np, edge_dist_np = build_knn_graph(coords, k=args.k, r_max=args.r_max)
    print(f"[INFO] TEST edges: {edge_index_np.shape[1]:,} (k={args.k}, r_max={args.r_max})")

    out_map, _ = parse_ccc_out_map(obs)
    ccc_e = build_ccc_edge_feat_for_knn(edge_index_np, out_map)  # (E,2)

    print(f"[INFO] CCC hit rate (test): out={ccc_e[:,0].mean():.4f}, in={ccc_e[:,1].mean():.4f}")

    # edge_feat = [dist, ccc_out, ccc_in]
    edge_feat_np = np.concatenate([edge_dist_np.reshape(-1,1), ccc_e], axis=1).astype(np.float32)  # (E,3)

    print("\n" + "="*70)
    print("STEP 6: Load CCC-attn GNN + refine")
    print("="*70)
    gnn_ckpt = torch.load(args.gnn_ckpt, map_location=device)

    P = int(gnn_ckpt.get("proj_dim", 256))
    alpha = float(gnn_ckpt.get("alpha", 1.0))
    cfg = gnn_ckpt.get("config", {})

    gnn_layers = int(cfg.get("gnn_layers", 3))
    gnn_hidden = int(cfg.get("gnn_hidden", 256))
    dropout = float(cfg.get("dropout", 0.0))

    D = cell_emb.shape[1]
    gene_enc = GenePredEncoder(G=G, out_dim=P, hidden=max(512, P*2), dropout=dropout).to(device)
    gnn = ResidualGNN(dim=D + P, hidden_dim=gnn_hidden, n_layers=gnn_layers, dropout=dropout, edge_dim=3).to(device)
    head = GeneDeltaHead(in_dim=D + P, hidden_dim=gnn_hidden, out_dim=G, dropout=dropout).to(device)

    # load weights (names match training)
    gene_enc.load_state_dict(gnn_ckpt["gene_enc"])
    gnn.load_state_dict(gnn_ckpt["gnn"])
    head.load_state_dict(gnn_ckpt["head"])

    gene_enc.eval(); gnn.eval(); head.eval()

    emb_t = torch.from_numpy(cell_emb).to(device)
    base_t = torch.from_numpy(pred_base_gene).to(device)

    ei_t = torch.from_numpy(edge_index_np).long().to(device)
    ef_t = torch.from_numpy(edge_feat_np).float().to(device)
    deg_t = compute_degrees(ei_t, N, device)

    with torch.no_grad():
        z = gene_enc(base_t)                 # (N,P)
        x = torch.cat([emb_t, z], dim=1)     # (N,D+P)
        x2 = gnn(x, ei_t, ef_t, deg_t)
        delta = head(x2)                     # (N,G)
        has_nb = (deg_t > 0).float().unsqueeze(-1)
        delta = delta * has_nb
        pred_refine_gene = (base_t + alpha * delta).detach().cpu().float().numpy().astype(np.float32)

    mse_ref, pcc_ref = masked_flat_metrics(pred_refine_gene, truth_gene, mask_gene)
    print(f"[REFINED][TEST] gene-space masked MSE: {mse_ref:.4f}")
    print(f"[REFINED][TEST] gene-space masked PCC: {pcc_ref:.4f}")
    print(f"[GAIN][TEST] ΔPCC = {pcc_ref - pcc_base:+.4f}")

    print("\n" + "="*70)
    print("STEP 7: Save h5ad (same structure as before)")
    print("="*70)
    var = pd.DataFrame(index=gene_names)
    out = ad.AnnData(X=np.zeros((N, G), dtype=np.float32), obs=obs, var=var)

    out.layers["pred_base"] = pred_base_gene.astype(np.float32)
    out.layers["pred_refine"] = pred_refine_gene.astype(np.float32)
    out.layers["truth_value"] = truth_gene.astype(np.float32)
    out.layers["mask_predicted_genes"] = mask_gene.astype(np.uint8)

    out.obsm["X_cell_emb"] = cell_emb.astype(np.float32)

    out.uns["test_eval"] = {
        "metric_space": "gene (masked by which genes were tokenized/predicted)",
        "base_pcc_masked": float(pcc_base),
        "refined_pcc_masked": float(pcc_ref),
        "base_mse_masked": float(mse_base),
        "refined_mse_masked": float(mse_ref),
        "k": int(args.k),
        "r_max": args.r_max,
        "alpha": float(alpha),
        "proj_dim": int(P),
        "gnn_ckpt": args.gnn_ckpt,
        "epif_ckpt": args.epif_ckpt,
        "edge_features": "dist + ccc_out(i->j) + ccc_in(j->i)",
        "note": "Inference matches CCC-attn training graph/features on kNN edges."
    }

    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    print(f"[INFO] Writing: {args.out_h5ad}")
    out.write_h5ad(args.out_h5ad)
    print("[DONE]")


if __name__ == "__main__":
    main()

