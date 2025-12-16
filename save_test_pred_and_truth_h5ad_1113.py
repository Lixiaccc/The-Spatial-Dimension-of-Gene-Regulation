#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import argparse
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import yaml

from model import EpiFoundation
from tokenizer import GeneVocab
from data.dataloader import PairedSCDataset


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # strip "module." only if present
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[INFO] load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}")
    return ckpt


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
    )
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)

    ap.add_argument("--rna_file", required=True)
    ap.add_argument("--atac_file", required=True)
    ap.add_argument("--rna_key", required=True, help="e.g. X_binned")
    ap.add_argument("--atac_key", required=True, help="e.g. X")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_h5ad", required=True)

    ap.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    ap.add_argument("--store_X", choices=["zeros", "pred", "truth"], default="zeros",
                    help="What to store in adata.X; layers always store pred/truth/mask.")
    args = ap.parse_args()

    config = load_yaml(args.config)
    train_config = config["train"]
    data_config = config["data"]
    vocab_config = config["vocab"]

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # vocabs
    print("[INFO] Loading vocabularies...")
    rna_vocab = GeneVocab.from_file(vocab_config["rna_path"])
    atac_vocab = GeneVocab.from_file(vocab_config["atac_path"])
    cell_vocab = GeneVocab.from_file(vocab_config["cell_type_path"])
    batch_vocab = GeneVocab.from_file(vocab_config["batch_path"])
    chr_vocab = GeneVocab.from_file(vocab_config["chr_path"])

    print("[INFO] RNA vocab size: ", len(rna_vocab))
    print("[INFO] ATAC vocab size:", len(atac_vocab))
    print("[INFO] CHR vocab size: ", len(chr_vocab))

    pad = vocab_config["special_tokens"]["pad"]
    cls = vocab_config["special_tokens"]["cls"]

    print("[INFO] Building PairedSCDataset (same tokenization as finetune)...")
    ds = build_paired_dataset(
        rna_file=args.rna_file,
        atac_file=args.atac_file,
        rna_key=args.rna_key,
        atac_key=args.atac_key,
        config=config,
        rna_vocab=rna_vocab,
        atac_vocab=atac_vocab,
        cell_vocab=cell_vocab,
        batch_vocab=batch_vocab,
        chr_vocab=chr_vocab,
    )
    gc.collect()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model
    print("[INFO] Creating model...")
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
    print("[INFO] Model loaded.")

    # collect
    all_pred_1d, all_truth_1d = [], []
    pred_rows, truth_rows, mask_rows = [], [], []

    print("[INFO] Running forward pass...")
    with torch.no_grad():
        for bi, batch in enumerate(loader, start=1):
            if bi == 1 or bi % 10 == 0 or bi == len(loader):
                print(f"[INFO] batch {bi}/{len(loader)}")

            rna_values = batch["rna_values"].to(device)   # (B, rna_max_len)
            rna_ids    = batch["rna_ids"].to(device)      # (B, rna_max_len)
            atac_ids   = batch["atac_ids"].to(device)
            batch_ids  = batch["batch_ids"].to(device)
            rna_chrs   = batch["rna_chrs"].to(device)
            atac_chrs  = batch["atac_chrs"].to(device)

            padding_positions = atac_ids.eq(atac_vocab[pad["token"]])
            rna_non_pad = rna_ids.ne(rna_vocab[pad["token"]])  # SAME mask as finetune

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                out = model(
                    atac=atac_ids,
                    rna=rna_ids,
                    src_key_padding_mask=padding_positions,
                    batch_id=batch_ids,
                    rna_chrs=rna_chrs,
                    atac_chrs=atac_chrs,
                )

            value_pred = out["value_pred"]  # (B, rna_max_len)

            pred_1d = value_pred[rna_non_pad].detach().cpu().float().numpy()
            tru_1d  = rna_values[rna_non_pad].detach().cpu().float().numpy()
            all_pred_1d.append(pred_1d)
            all_truth_1d.append(tru_1d)

            pred_rows.append(value_pred.detach().cpu().float().numpy())
            truth_rows.append(rna_values.detach().cpu().float().numpy())
            mask_rows.append(rna_non_pad.detach().cpu().numpy().astype(np.uint8))

    all_pred_1d = np.concatenate(all_pred_1d, axis=0)
    all_truth_1d = np.concatenate(all_truth_1d, axis=0)

    masked_mse = float(np.mean((all_pred_1d - all_truth_1d) ** 2))
    masked_pcc = pearson_safe(all_pred_1d, all_truth_1d)

    pred_mat = np.concatenate(pred_rows, axis=0).astype(np.float32)    # (cells, rna_max_len)
    truth_mat = np.concatenate(truth_rows, axis=0).astype(np.float32)  # (cells, rna_max_len)
    mask_mat = np.concatenate(mask_rows, axis=0).astype(np.uint8)      # (cells, rna_max_len)

    print("\n" + "=" * 70)
    print("EVAL METRICS (token-space, masked; SHOULD match finetune PCC)")
    print("=" * 70)
    print("Matrix shape (cells x rna_max_len):", pred_mat.shape)
    print("Masked MSE:", round(masked_mse, 4))
    print("Masked PCC(flat):", round(float(masked_pcc), 4))
    print("=" * 70)

    n_cells, rna_max_len = pred_mat.shape

    # obs: keep ATAC obs if available (so barcodes/coords survive)
    try:
        obs = ds.atac_adata.obs.copy()
    except Exception:
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])

    # IMPORTANT: var must have 3000 entries so layers can be (488,3000)
    var = pd.DataFrame(index=[f"tok_{i}" for i in range(rna_max_len)])

    if args.store_X == "pred":
        X = pred_mat
    elif args.store_X == "truth":
        X = truth_mat
    else:
        X = np.zeros((n_cells, rna_max_len), dtype=np.float32)

    out = ad.AnnData(X=X, obs=obs, var=var)
    out.layers["pred_value"] = pred_mat
    out.layers["truth_value"] = truth_mat
    out.layers["mask"] = mask_mat

    out.uns["eval_info"] = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "rna_file": args.rna_file,
        "atac_file": args.atac_file,
        "rna_key": args.rna_key,
        "atac_key": args.atac_key,
        "masked_mse_flat": masked_mse,
        "masked_pcc_flat": masked_pcc,
        "note": "Token-space eval. Columns are token positions tok_0..tok_{rna_max_len-1} including CLS at tok_0.",
    }

    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    print(f"[INFO] Saving eval-ready h5ad to: {args.out_h5ad}")
    out.write_h5ad(args.out_h5ad)
    print("[DONE]")


if __name__ == "__main__":
    main()

