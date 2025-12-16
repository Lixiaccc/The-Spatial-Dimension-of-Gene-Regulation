# -*- coding: utf-8 -*-

import os
import gc
import argparse
import json
import random
import math

import numpy as np
import pandas as pd
from scipy import sparse

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from tensorboardX import SummaryWriter

from model import EpiFoundation
from loss.loss import MaskedMSELoss
from data.dataloader import PairedSCDataset
from tokenizer import GeneVocab
from utils import *
from memory_profiler import profile

import yaml
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="./configs/ours/finetune_ours_3000hvg.yml",
    help="Config file.",
)
args = parser.parse_args()


def main():
    # ------------------------------------------------------------------
    # 0. Distributed setup
    # ------------------------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()
    is_master = local_rank == 0

    # ------------------------------------------------------------------
    # 1. Read and parse config
    # ------------------------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_config = config["train"]
    valid_config = config["valid"]
    data_config = config["data"]
    vocab_config = config["vocab"]
    task_name = config["task_name"]

    task_folder = f"./experiment/{task_name}"
    ckpt_dir = os.path.join(task_folder, "ckpts")
    os.makedirs(task_folder, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    random_seed = train_config["seed"]
    EPOCHS = train_config["epochs"]
    BATCH_SIZE = train_config["batch_size"]
    GRADIENT_ACCUMULATION = train_config["gradient_accumulation_steps"]
    LEARNING_RATE = float(train_config["lr"])
    model_name = train_config["model"]["encoder"]

    save_ckpt_freq = train_config.get("save_ckpt_freq", 5)
    resume = train_config.get("resume", False)

    # special tokens
    pad = vocab_config["special_tokens"]["pad"]
    mask = vocab_config["special_tokens"]["mask"]
    cls = vocab_config["special_tokens"]["cls"]

    # set seeds
    seed_all(random_seed + dist.get_rank())

    # init loggers
    logger = set_log(log_dir=os.path.join(task_folder, "logs"))
    tb_logger = SummaryWriter(os.path.join(task_folder, "tb_logs"))

    if is_master:
        logger.info("===== CONFIG =====")
        logger.info(dict2str(config))

    # ------------------------------------------------------------------
    # 2. Load vocabs
    # ------------------------------------------------------------------
    rna_vocab = GeneVocab.from_file(vocab_config["rna_path"])
    atac_vocab = GeneVocab.from_file(vocab_config["atac_path"])
    cell_vocab = GeneVocab.from_file(vocab_config["cell_type_path"])
    batch_vocab = GeneVocab.from_file(vocab_config["batch_path"])
    chr_vocab = GeneVocab.from_file(vocab_config["chr_path"])

    if is_master:
        logger.info(f"RNA vocab size:   {len(rna_vocab)}")
        logger.info(f"ATAC vocab size:  {len(atac_vocab)}")
        logger.info(f"Cell vocab size:  {len(cell_vocab)}")
        logger.info(f"Batch vocab size: {len(batch_vocab)}")
        logger.info(f"Chr vocab size:   {len(chr_vocab)}")

    # ------------------------------------------------------------------
    # 3. Datasets & Dataloaders
    # ------------------------------------------------------------------
    if is_master:
        logger.info("===== Loading TRAIN set =====")
    train_set = PairedSCDataset(
        rna_file=data_config["train"]["rna_path"],
        atac_file=data_config["train"]["atac_path"],
        rna_key=data_config["train"]["rna_key"],
        atac_key=data_config["train"]["atac_key"],
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
        logger=logger,
    )
    gc.collect()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
    )
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        prefetch_factor=4,
        num_workers=4,
    )

    if is_master:
        logger.info("===== Loading VAL set =====")
    val_set = PairedSCDataset(
        rna_file=data_config["val"]["rna_path"],
        atac_file=data_config["val"]["atac_path"],
        rna_key=data_config["val"]["rna_key"],
        atac_key=data_config["val"]["atac_key"],
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
        logger=logger,
    )
    gc.collect()
    val_sampler = DistributedSampler(
        val_set, num_replicas=world_size, rank=dist.get_rank(), shuffle=False
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        prefetch_factor=4,
        num_workers=4,
    )

    # ------------------------------------------------------------------
    # 4. Create model (value_finetune only)
    # ------------------------------------------------------------------
    if is_master:
        logger.info("===== Creating model =====")

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
        encoder=model_name,
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

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=LEARNING_RATE,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9,
    )

    start_epoch = 1
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    scaler = torch.cuda.amp.GradScaler(enabled=train_config["amp"])

    mvc_loss_fn = MaskedMSELoss().to(device)
    mvc_weight = train_config["task_weight"]["mvc"]
    # cell_type_weight = train_config["task_weight"]["cell_type"]  # keep but unused
    softmax = nn.Softmax(dim=-1)

    steps = 0

    # ------------------------------------------------------------------
    # 5. Load pretrained checkpoint (encoder weights only)
    # ------------------------------------------------------------------
    if train_config["model"]["pretrained"] is not None:
        if is_master:
            logger.info(
                f"Loading pretrained model from: {train_config['model']['pretrained']}"
            )
        checkpoint = torch.load(
            train_config["model"]["pretrained"], map_location=device
        )
        
        pretrained_dict = checkpoint["model"]
        model_dict = model.module.state_dict()
        
        # Load only compatible layers
        loaded_keys = []
        skipped_keys = []
        
        for k, v in pretrained_dict.items():
            # Skip decoders
            if "value_decoder" in k or "mvc_decoder" in k or "cls_decoder" in k:
                skipped_keys.append(k)
                continue
            
            # Skip mismatched encoder layers
            if "encoder.transformer_encoder" in k and k not in model_dict:
                skipped_keys.append(k)
                continue
            
            # Load if shapes match
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)
        
        model.module.load_state_dict(model_dict, strict=False)
        
        if is_master:
            logger.info(f"✓ Loaded {len(loaded_keys)} compatible parameters")
            logger.info(f"✗ Skipped {len(skipped_keys)} incompatible parameters")



        if resume:
            start_epoch = checkpoint["epoch"] + 1
            steps = checkpoint["steps"]
        del checkpoint
        del pretrained_dict
        gc.collect()

    dist.barrier()
    if is_master:
        logger.info(
            f"Start finetuning from epoch: {start_epoch}, global steps: {steps}"
        )

    # ------------------------------------------------------------------
    # 6. Training loop with early stopping
    # ------------------------------------------------------------------
    patience = train_config["early_stop"]["patience"]
    min_delta = train_config["early_stop"]["min_delta"]
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    train_epoch_losses = []
    val_epoch_losses = []

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        # --- TRAIN ---
        train_sampler.set_epoch(epoch)
        model.train()
        dist.barrier()

        running_loss = 0.0
        steps_in_epoch = 0

        if is_master:
            logger.info(
                f"Training epoch {epoch} with {len(train_loader.dataset)} samples, {len(train_loader)} steps"
            )

        for batch_idx, batch in enumerate(train_loader, start=1):
            steps += 1
            steps_in_epoch += 1

            rna_values = batch["rna_values"].to(device)
            rna_ids = batch["rna_ids"].to(device)
            atac_ids = batch["atac_ids"].to(device)
            cell_ids = batch["cell_ids"].to(device)
            batch_ids = batch["batch_ids"].to(device)
            rna_chrs = batch["rna_chrs"].to(device)
            atac_chrs = batch["atac_chrs"].to(device)

            padding_positions = atac_ids.eq(atac_vocab[pad["token"]])
            rna_non_pad = rna_ids.ne(rna_vocab[pad["token"]])

            is_accum_step = (
                batch_idx % GRADIENT_ACCUMULATION != 0
                and batch_idx != len(train_loader)
            )

            autocast_ctx = torch.cuda.amp.autocast(
                enabled=train_config["amp"], dtype=torch.bfloat16
            )

            if is_accum_step:
                with model.no_sync():
                    with autocast_ctx:
                        out = model(
                            atac=atac_ids,
                            rna=rna_ids,
                            src_key_padding_mask=padding_positions,
                            batch_id=batch_ids,
                            rna_chrs=rna_chrs,
                            atac_chrs=atac_chrs,
                        )
                        # ✅ use ONLY value_pred for our regression loss
                        value_pred = out["value_pred"]
                        mvc_loss = (
                            mvc_loss_fn(
                                value_pred, rna_values.float(), mask=rna_non_pad
                            )
                            * mvc_weight
                        )
                        loss = mvc_loss
                        running_loss += loss.item()
                        loss = loss / GRADIENT_ACCUMULATION
                    scaler.scale(loss).backward()
            else:
                with autocast_ctx:
                    out = model(
                        atac=atac_ids,
                        rna=rna_ids,
                        src_key_padding_mask=padding_positions,
                        batch_id=batch_ids,
                        rna_chrs=rna_chrs,
                        atac_chrs=atac_chrs,
                    )
                    value_pred = out["value_pred"]
                    mvc_loss = (
                        mvc_loss_fn(
                            value_pred, rna_values.float(), mask=rna_non_pad
                        )
                        * mvc_weight
                    )
                    loss = mvc_loss
                    running_loss += loss.item()
                    if is_master:
                        tb_logger.add_scalar(
                            "train/mvc_loss", mvc_loss.item(), steps
                        )
                        logger.info(
                            f"Epoch {epoch} | Step {batch_idx} | MVC Loss: {mvc_loss:.4f}"
                        )
                    loss = loss / GRADIENT_ACCUMULATION
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # average loss over steps
        epoch_train_loss = running_loss / max(steps_in_epoch, 1)
        epoch_train_loss = get_reduced(
            epoch_train_loss, local_rank, 0, world_size
        )
        train_epoch_losses.append(float(epoch_train_loss))

        if is_master:
            logger.info(
                f"Epoch {epoch} | Train MVC Loss (avg): {epoch_train_loss:.4f}"
            )
            tb_logger.add_scalar("epoch/train_mvc_loss", epoch_train_loss, epoch)

        scheduler.step()

        # --- VALIDATION ---
        if epoch % valid_config["freq"] == 0:
            if is_master:
                logger.info("#### Validation ####")
            model.eval()
            dist.barrier()

            val_running_loss = 0.0
            val_steps = 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader, start=1):
                    val_steps += 1
                    rna_values = batch["rna_values"].to(device)
                    rna_ids = batch["rna_ids"].to(device)
                    atac_ids = batch["atac_ids"].to(device)
                    cell_ids = batch["cell_ids"].to(device)
                    batch_ids = batch["batch_ids"].to(device)
                    rna_chrs = batch["rna_chrs"].to(device)
                    atac_chrs = batch["atac_chrs"].to(device)

                    padding_positions = atac_ids.eq(atac_vocab[pad["token"]])
                    rna_non_pad = rna_ids.ne(rna_vocab[pad["token"]])

                    with torch.cuda.amp.autocast(
                        enabled=train_config["amp"], dtype=torch.bfloat16
                    ):
                        out = model(
                            atac=atac_ids,
                            rna=rna_ids,
                            src_key_padding_mask=padding_positions,
                            batch_id=batch_ids,
                            rna_chrs=rna_chrs,
                            atac_chrs=atac_chrs,
                        )
                        value_pred = out["value_pred"]
                        mvc_loss = (
                            mvc_loss_fn(
                                value_pred,
                                rna_values.float(),
                                mask=rna_non_pad,
                            )
                            * mvc_weight
                        )
                    val_running_loss += mvc_loss.item()
                    
                    # Collect for PCC
                    pred_np = value_pred[rna_non_pad].detach().cpu().float().numpy()
                    target_np = rna_values[rna_non_pad].cpu().float().numpy()
                    all_preds.extend(pred_np.tolist())
                    all_targets.extend(target_np.tolist())

            epoch_val_loss = val_running_loss / max(val_steps, 1)
            epoch_val_loss = get_reduced(
                epoch_val_loss, local_rank, 0, world_size
            )
            val_epoch_losses.append(float(epoch_val_loss))

            # Calculate PCC
            val_pcc = pearsonr(all_preds, all_targets)[0] if len(all_preds) > 0 else 0.0
            
            if is_master:
                logger.info(
                    f"Epoch {epoch} | Train MSE: {epoch_train_loss:.4f} | Val MSE: {epoch_val_loss:.4f} | Val PCC: {val_pcc:.4f}"
                )
                tb_logger.add_scalar(
                    "epoch/val_mvc_loss", epoch_val_loss, epoch
                )

            # --- Early stopping & best ckpt ---
            if epoch_val_loss + min_delta < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                patience_counter = 0
                if is_master:
                    logger.info(
                        f"✓ NEW BEST Val MSE: {best_val_loss:.4f} at epoch {epoch}"
                    )
                    save_ckpt(
                        epoch,
                        steps,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch_val_loss,
                        task_name,
                        ckpt_dir,
                    )
            else:
                patience_counter += 1
                if is_master:
                    logger.info(
                        f"No improvement in val loss. Patience {patience_counter}/{patience}"
                    )
                if patience_counter >= patience:
                    if is_master:
                        logger.info(
                            f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} with val loss {best_val_loss:.4f}"
                        )
                    break

        # periodic ckpt (not necessarily best)
        if is_master and (epoch % save_ckpt_freq == 0):
            save_ckpt(
                epoch,
                steps,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch_train_loss,
                task_name,
                ckpt_dir,
            )

    # ------------------------------------------------------------------
    # 7. Save loss curves
    # ------------------------------------------------------------------
    if is_master:
        losses_path = os.path.join(task_folder, "train_val_losses.csv")
        df_losses = pd.DataFrame(
            {
                "epoch": list(range(start_epoch, start_epoch + len(train_epoch_losses))),
                "train_mvc_loss": train_epoch_losses,
                "val_mvc_loss": val_epoch_losses
                + [np.nan]
                * (len(train_epoch_losses) - len(val_epoch_losses)),
            }
        )
        df_losses.to_csv(losses_path, index=False)

        plt.figure()
        plt.plot(
            range(start_epoch, start_epoch + len(train_epoch_losses)),
            train_epoch_losses,
            label="Train MVC loss",
        )
        if len(val_epoch_losses) > 0:
            plt.plot(
                range(start_epoch, start_epoch + len(val_epoch_losses)),
                val_epoch_losses,
                label="Val MVC loss",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(task_folder, "loss_curve.png")
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"Saved train/val losses to: {losses_path}")
        logger.info(f"Saved loss curve PNG to: {fig_path}")


if __name__ == "__main__":
    main()

