# evaluate_test_accuracy.py
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import yaml

from tokenizer import GeneVocab
from model import EpiFoundation
from data.dataloader import PairedSCDataset

def pcc(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return pearsonr(a, b)[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)

    # Option 1: use config split (train/val/test) if present
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])

    # Option 2: override files/keys explicitly (works even if config has no test)
    ap.add_argument("--rna_file", default=None)
    ap.add_argument("--atac_file", default=None)
    ap.add_argument("--rna_key", default=None)   # e.g. X_binned
    ap.add_argument("--atac_key", default=None)  # e.g. counts

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_config = config["train"]
    data_config  = config["data"]
    vocab_config = config["vocab"]

    pad = vocab_config["special_tokens"]["pad"]
    cls = vocab_config["special_tokens"]["cls"]

    # -------- choose data source --------
    if args.rna_file is not None and args.atac_file is not None:
        rna_file  = args.rna_file
        atac_file = args.atac_file
        rna_key   = args.rna_key  if args.rna_key  is not None else data_config["train"]["rna_key"]
        atac_key  = args.atac_key if args.atac_key is not None else data_config["train"]["atac_key"]
        print(f"[INFO] Using OVERRIDE files (rna={rna_file}, atac={atac_file})")
        print(f"[INFO] Using keys: rna_key={rna_key}, atac_key={atac_key}")
    else:
        if args.split not in data_config:
            raise KeyError(
                f"Config data has no '{args.split}' block. "
                f"Either add data:{args.split}:... to YAML, or pass --rna_file/--atac_file overrides."
            )
        rna_file  = data_config[args.split]["rna_path"]
        atac_file = data_config[args.split]["atac_path"]
        rna_key   = data_config[args.split]["rna_key"]
        atac_key  = data_config[args.split]["atac_key"]
        print(f"[INFO] Using config split='{args.split}': rna={rna_file}, atac={atac_file}")

    # -------- vocabs --------
    rna_vocab   = GeneVocab.from_file(vocab_config["rna_path"])
    atac_vocab  = GeneVocab.from_file(vocab_config["atac_path"])
    cell_vocab  = GeneVocab.from_file(vocab_config["cell_type_path"])
    batch_vocab = GeneVocab.from_file(vocab_config["batch_path"])
    chr_vocab   = GeneVocab.from_file(vocab_config["chr_path"])

    # -------- dataset (same as finetune) --------
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

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -------- model --------
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

    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt["model"]
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    model.load_state_dict(sd, strict=False)
    model.eval()
    print("[INFO] Loaded checkpoint:", args.checkpoint)

    all_pred_flat = []
    all_true_flat = []
    pred_mat = []
    true_mat = []

    use_amp = (device.type == "cuda") and (not args.no_amp)

    with torch.no_grad():
        for batch in loader:
            rna_values = batch["rna_values"].to(device).float()
            rna_ids    = batch["rna_ids"].to(device)
            atac_ids   = batch["atac_ids"].to(device)
            batch_ids  = batch["batch_ids"].to(device)
            rna_chrs   = batch["rna_chrs"].to(device)
            atac_chrs  = batch["atac_chrs"].to(device)

            padding_positions = atac_ids.eq(atac_vocab[pad["token"]])

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                out = model(
                    atac=atac_ids,
                    rna=rna_ids,
                    src_key_padding_mask=padding_positions,
                    batch_id=batch_ids,
                    rna_chrs=rna_chrs,
                    atac_chrs=atac_chrs,
                )

            value_pred = out["value_pred"].float()

            # match finetune's mask style, but EXCLUDE CLS explicitly
            nonpad = rna_ids.ne(rna_vocab[pad["token"]])
            nonpad[:, 0] = False  # drop CLS

            all_pred_flat.append(value_pred[nonpad].detach().cpu().numpy())
            all_true_flat.append(rna_values[nonpad].detach().cpu().numpy())

            pred_mat.append(value_pred[:, 1:].detach().cpu().numpy())
            true_mat.append(rna_values[:, 1:].detach().cpu().numpy())

    all_pred_flat = np.concatenate(all_pred_flat)
    all_true_flat = np.concatenate(all_true_flat)

    pred_mat = np.concatenate(pred_mat, axis=0)  # (cells, seq_len-1)
    true_mat = np.concatenate(true_mat, axis=0)

    mse = float(np.mean((pred_mat - true_mat) ** 2))
    flat = pcc(all_pred_flat, all_true_flat)

    cell_pcc = np.array([pcc(pred_mat[i, :], true_mat[i, :]) for i in range(pred_mat.shape[0])], dtype=float)
    gene_pcc = np.array([pcc(pred_mat[:, j], true_mat[:, j]) for j in range(pred_mat.shape[1])], dtype=float)

    print("\n" + "=" * 70)
    print("TEST ACCURACY (SAME TOKENIZATION AS FINETUNE)")
    print("=" * 70)
    print("Matrix shape (cells x tokens_without_CLS):", pred_mat.shape)
    print("MSE:", round(mse, 4))
    print("PCC(flat like finetune):", round(float(flat), 4))
    print("PCC(per-cell): mean", round(float(np.nanmean(cell_pcc)), 4),
          "median", round(float(np.nanmedian(cell_pcc)), 4))
    print("PCC(per-gene): mean", round(float(np.nanmean(gene_pcc)), 4),
          "median", round(float(np.nanmedian(gene_pcc)), 4))
    print("=" * 70)

if __name__ == "__main__":
    main()

