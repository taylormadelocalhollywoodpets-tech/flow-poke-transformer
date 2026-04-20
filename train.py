# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2026 Stefan Baumann et al., CompVis @ LMU Munich

import os
import json
import math
from pathlib import Path
import logging
import random
from datetime import datetime

import click
import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention.flex_attention import create_block_mask
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LambdaLR, SequentialLR
import numpy as np
from tqdm.auto import tqdm
from einops import rearrange, repeat


def endless_iter(iterable):
    while True:
        yield from iterable


def make_scheduler(optimizer, lr, warmup_steps, max_steps, scheduler_type="linear"):
    """Build a LR scheduler.

    scheduler_type="linear":  linear warmup → linear decay to 0
    scheduler_type="cosine":  linear warmup → cosine annealing to 1e-8
    """
    has_warmup = warmup_steps > 0
    has_decay = max_steps is not None

    if scheduler_type == "cosine":
        if has_warmup and has_decay:
            warmup = LinearLR(optimizer, start_factor=1e-8 / lr, end_factor=1.0, total_iters=warmup_steps)
            decay = CosineAnnealingLR(optimizer, T_max=max(1, max_steps - warmup_steps), eta_min=1e-8)
            return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])
        elif has_warmup:
            return LinearLR(optimizer, start_factor=1e-8 / lr, end_factor=1.0, total_iters=warmup_steps)
        elif has_decay:
            return CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-8)
        else:
            return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:  # linear
        if has_warmup and has_decay:
            warmup = LinearLR(optimizer, start_factor=1e-8 / lr, end_factor=1.0, total_iters=warmup_steps)
            decay = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max(1, max_steps - warmup_steps))
            return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])
        elif has_warmup:
            return LinearLR(optimizer, start_factor=1e-8 / lr, end_factor=1.0, total_iters=warmup_steps)
        elif has_decay:
            return LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_steps)
        else:
            return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)



# ---------------------------------------------------------------------------
# Model-specific step protocols
# Each factory receives (model, real_model, device, device_type, is_distributed)
# and returns (init_caches, compute_step).
# ---------------------------------------------------------------------------

def fpt_make_train_fns(model, real_model, device, device_type, is_distributed):
    """Step protocol for the original FlowPokeTransformer (flow_poke.model)."""
    from flow_poke.model import query_causal_mask_mod as fpt_mask_mod

    def init_caches(batch):
        B = batch["pos_poke"].size(0)
        L_poke = batch["pos_poke"].size(1)
        L_query = math.prod(batch["pos_query"].shape[1:3])
        block_mask = create_block_mask(
            fpt_mask_mod(sequence_length=L_poke, n_query=batch["pos_query"].size(2)),
            B=1, H=1,
            Q_LEN=L_poke + L_query,
            KV_LEN=L_poke + L_query,
            device=device,
        )
        is_query = repeat(
            torch.cat([
                torch.zeros(L_poke, dtype=torch.bool, device=device),
                torch.ones(L_query, dtype=torch.bool, device=device),
            ]),
            "l -> b l", b=B,
        )
        return block_mask, is_query, L_poke

    def compute_step(batch, block_mask, is_query, L_poke, compute_metrics=False, flow_mask=None):
        pos = torch.cat([
            batch["pos_poke"],
            rearrange(batch["pos_query"], "b n_p n_q c -> b (n_p n_q) c"),
        ], dim=1)
        flow_target = rearrange(batch["flow_query"], "b n_p n_q c -> b (n_p n_q) c")
        flow = torch.cat([batch["flow_poke"], torch.zeros_like(flow_target)], dim=1)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            d_img = real_model.embed_image(batch["x"])
            distribution = model(
                pos=pos, flow=flow, is_query=is_query,
                camera_static=batch["camera_static"],
                mask=block_mask, d_img=d_img,
            )

        loss = -distribution[:, L_poke:].log_prob(flow_target).mean()

        if not compute_metrics:
            return loss

        with torch.no_grad():
            metrics = {}
            samples = distribution[:, L_poke:].sample()
            epe = (samples - flow_target).norm(p=2, dim=-1)
            metrics["epe"] = epe.mean().detach()
            for alpha in [0.1, 0.01, 0.001]:
                metrics[f"pck@{alpha}"] = epe.le(alpha).float().mean().detach()
            metrics["flow_mag_gt"] = flow_target.norm(p=2, dim=-1).mean().detach()
            metrics["flow_mag_pred"] = samples.norm(p=2, dim=-1).mean().detach()
            metrics["frac_static_camera"] = batch["camera_static"].float().mean().detach()

        return loss, metrics

    return init_caches, compute_step


def myriad_make_train_fns(model, real_model, device, device_type, is_distributed):
    """Step protocol for the Myriad MyriadStepByStep models (myriad.model)."""
    from myriad.model import query_causal_mask_mod as myriad_mask_mod

    def init_caches(batch):
        B = batch["pos_poke"].size(0)
        L_poke = batch["pos_poke"].size(1)
        L_query = batch["pos_query"].size(1)
        assert L_query == L_poke, f"Expected L_query == L_poke, got {L_query} vs {L_poke}"

        d_img = real_model.embed_image(batch["x"].to(device))
        L_prefix = d_img["L"]
        block_mask = create_block_mask(
            myriad_mask_mod(l_prefix=L_prefix, l_seq=L_poke, n_query=1),
            B=1, H=1,
            Q_LEN=L_prefix + L_poke + L_query,
            KV_LEN=L_prefix + L_poke + L_query,
            device=device,
        )
        is_query = repeat(
            torch.cat([
                torch.zeros(L_poke, dtype=torch.bool, device=device),
                torch.ones(L_query, dtype=torch.bool, device=device),
            ]),
            "l -> b l", b=B,
        )
        return block_mask, is_query, L_poke

    def compute_step(batch, block_mask, is_query, L_poke, compute_metrics=False, flow_mask=None):
        pos_poke, pos_query = batch["pos_poke"], batch["pos_query"]
        if pos_query.ndim > pos_poke.ndim:
            pos_query = rearrange(pos_query, "b n_p n_q ... -> b (n_p n_q) ...")
        pos = torch.cat([pos_poke, pos_query], dim=1)

        pos_orig = (
            torch.cat([batch["pos_orig_poke"], batch["pos_orig_query"]], dim=1)
            if "pos_orig_poke" in batch and "pos_orig_query" in batch else None
        )
        t = (
            torch.cat([batch["t_poke"], batch["t_query"]], dim=1)
            if "t_poke" in batch and "t_query" in batch else None
        )
        track_id = (
            torch.cat([batch["id_poke"], batch["id_query"]], dim=1)
            if "id_poke" in batch and "id_query" in batch else None
        )

        flow_poke, flow_query = batch["flow_poke"], batch["flow_query"]
        if flow_query.ndim > flow_poke.ndim:
            flow_query = rearrange(flow_query, "b n_p n_q ... -> b (n_p n_q) ...")
        flow_target = flow_query
        flow = torch.cat([flow_poke, torch.zeros_like(flow_target)], dim=1)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            d_img = real_model.embed_image(batch["x"])
            distribution = model(
                pos=pos, pos_orig=pos_orig, t=t, track_id=track_id,
                flow=flow, is_query=is_query,
                camera_static=batch["camera_static"],
                mask=block_mask, d_img=d_img, track_id_emb_table=None,
            )
            loss = distribution[:, L_poke:].loss(flow_target)
            if flow_mask is not None:
                valid = flow_mask.sum()
                loss = (loss * flow_mask).sum() / valid if valid > 0 else loss.new_zeros(())
            else:
                loss = loss.mean()

        if not compute_metrics:
            return loss

        with torch.no_grad():
            metrics = {}
            samples = distribution[:, L_poke:].sample()
            epe = (samples - flow_target).norm(dim=-1)
            metrics["epe"] = epe.mean().detach()
            for alpha in [0.1, 0.01, 0.001]:
                metrics[f"pck@{alpha}"] = epe.le(alpha).float().mean().detach()
            metrics["flow_mag_gt"] = flow_target.norm(p=2, dim=-1).mean().detach()
            metrics["flow_mag_pred"] = samples.norm(p=2, dim=-1).mean().detach()
            metrics["frac_static_camera"] = batch["camera_static"].float().mean().detach()

        return loss, metrics

    return init_caches, compute_step


# ---------------------------------------------------------------------------
# Shared training infrastructure
# ---------------------------------------------------------------------------

def _train(
    data,
    model_cls,
    make_train_fns,
    out_dir,
    max_steps,
    checkpoint_freq,
    clip_grad_norm,
    compile,
    autotune,
    load_checkpoint,
    ckpt_load_optim,
    ckpt_load_scheduler,
    lr,
    weight_decay,
    warmup_steps,
    scheduler_type,
    wandb_enabled,
    wandb_project,
    config_dict,
):
    # Output & logging setup
    slurm_id = os.environ.get("SLURM_JOB_ID")
    timestamp = datetime.now().strftime("%H-%M-%S")
    date_str = datetime.now().strftime("%Y-%m-%d")
    run_id = slurm_id if slurm_id is not None else timestamp
    out_path = Path(out_dir) / date_str / run_id
    out_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_path / "train.log"),
        ],
    )

    # Distributed init & single-GPU fallback
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group()
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device_type = "cuda"
        device = torch.device(f"{device_type}:{local_rank}")
        torch.cuda.set_device(device)
        logger.info(f"Running distributed. Local rank: {local_rank}, World size: {world_size}")
        rank0logger = logging.getLogger(__name__)
        if rank != 0:
            rank0logger.disabled = True
        barrier = dist.barrier
    else:
        rank = 0
        device_type = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)
        logger.info(f"Running non-distributed on {device_type}")
        rank0logger = logger
        barrier = lambda: None

    # Save config
    if rank == 0:
        config_path = out_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        rank0logger.info(f"Saved config to {config_path}")

    # WandB setup
    if wandb_enabled and rank == 0:
        import wandb
        wandb.init(
            project=wandb_project,
            config=config_dict | {"global_batch_size": config_dict.get("batch_size", 1) * world_size},
            dir=out_path,
        )

    # Checkpoint loading pt1: read step counter before seeding
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint, weights_only=False, map_location=device)
        start_step = checkpoint["step"]
        rank0logger.info(f"Loaded checkpoint from {load_checkpoint} @ step {start_step}.")
    else:
        checkpoint = None
        start_step = 0

    # Seeding
    seed = 42 + rank + start_step
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = model_cls().to(device)
    real_model = model.module if hasattr(model, "module") else model
    optimizer = AdamW(real_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(
        optimizer, lr=lr,
        warmup_steps=warmup_steps, max_steps=max_steps,
        scheduler_type=scheduler_type,
    )

    rank0logger.info(model)
    rank0logger.info(
        f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M"
        f" ({sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M trainable)"
    )

    # Checkpoint loading pt2: restore state
    if load_checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        if ckpt_load_optim:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if ckpt_load_scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        rank0logger.info("Checkpoint state loaded.")

    # DDP wrapping (after state load so we wrap the restored model)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], static_graph=True)  # type: ignore

    # Build model-specific step functions
    init_caches, compute_step = make_train_fns(model, real_model, device, device_type, is_distributed)

    if compile:
        compute_step = torch.compile(
            compute_step, fullgraph=False, mode="max-autotune" if autotune else "default"
        )
        rank0logger.info("Step function compiled with torch.compile.")

    barrier()

    train_loader = data.train_dataloader()

    # Training loop
    # Caches (block_mask, is_query, L_poke) are initialized once from the first batch,
    # assuming consistent batch shapes throughout training.
    caches_initialized = False
    block_mask = is_query = L_poke = None

    if rank == 0:
        logger.info("Starting training...")

    done = False
    for i, batch in enumerate(
        pbar := tqdm(endless_iter(train_loader), desc="Training", disable=rank != 0, initial=start_step)
    ):
        try:
            if not caches_initialized:
                block_mask, is_query, L_poke = init_caches(batch)
                caches_initialized = True

            flow_mask = batch.get("flow_loss_mask", None)
            optimizer.zero_grad()
            loss, metrics = compute_step(
                {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()},
                block_mask=block_mask,
                is_query=is_query,
                L_poke=L_poke,
                compute_metrics=True,
                flow_mask=flow_mask.to(device) if flow_mask is not None else None,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            scheduler.step()

            avg_loss = (
                dist_nn.all_reduce(loss.detach().clone(), op=dist.ReduceOp.SUM) / world_size
                if is_distributed else loss.detach()
            )
            metrics = {
                k: (
                    dist_nn.all_reduce(v.detach(), op=dist.ReduceOp.SUM) / world_size
                    if is_distributed else v.detach()
                ).item()
                for k, v in metrics.items()
            }
            train_meta = {
                "loss": avg_loss.item(),
                "grad_norm": grad_norm.item(),
                "lr": scheduler.get_last_lr()[0],
            } | metrics

            pbar.set_postfix(train_meta)
            if wandb_enabled and rank == 0:
                import wandb
                wandb.log({f"train/{k}": v for k, v in train_meta.items()}, step=start_step + i)

            done = max_steps is not None and (start_step + i) >= max_steps
            if done:
                rank0logger.info(f"Reached max steps: {start_step + i} >= {max_steps}. Stopping training...")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping training...")
            done = True

        if done or (i % checkpoint_freq == 0 and rank == 0 and i > 0):
            checkpoint = {
                "model": real_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": start_step + i,
            }
            ckpt_dir = out_path / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, ckpt_dir / f"checkpoint_{start_step + i:07}.pt")
            rank0logger.info(f"Saved checkpoint at step {start_step + i}.")

        if done:
            break

    barrier()
    rank0logger.info("Training stopped.")


# ---------------------------------------------------------------------------
# Common CLI options shared by all subcommands
# ---------------------------------------------------------------------------

def common_options(func):
    options = [
        click.option("--out-dir", default="outputs", show_default=True, help="Root output directory."),
        click.option("--max-steps", default=None, type=int, help="Stop after this many steps (None = run forever)."),
        click.option("--checkpoint-freq", default=25_000, show_default=True, help="Save a checkpoint every N steps."),
        click.option("--clip-grad-norm", default=1.0, show_default=True, type=float, help="Gradient clipping norm."),
        click.option("--compile/--no-compile", default=False, show_default=True, help="Use torch.compile."),
        click.option("--autotune/--no-autotune", default=False, show_default=True, help="Use max-autotune mode."),
        click.option("--load-checkpoint", default=None, type=click.Path(exists=True), help="Checkpoint to resume from."),
        click.option("--ckpt-load-optim/--no-ckpt-load-optim", default=True, show_default=True, help="Restore optimizer state."),
        click.option("--ckpt-load-scheduler/--no-ckpt-load-scheduler", default=True, show_default=True, help="Restore scheduler state."),
        click.option("--lr", default=1e-4, show_default=True, type=float, help="Peak learning rate."),
        click.option("--weight-decay", default=1e-2, show_default=True, type=float, help="AdamW weight decay."),
        click.option("--warmup-steps", default=1_000, show_default=True, type=int, help="Linear LR warmup steps."),
        click.option("--scheduler", "scheduler_type", default="linear", show_default=True,
                     type=click.Choice(["linear", "cosine"]), help="LR decay schedule after warmup."),
        click.option("--wandb/--no-wandb", "wandb_enabled", default=False, show_default=True, help="Enable W&B logging."),
        click.option("--wandb-project", default="flow-poke-reasoner", show_default=True, help="W&B project name."),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


# ---------------------------------------------------------------------------
# CLI group & subcommands
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """Flow Poke Transformer / Myriad training."""
    pass


@cli.command("fpt")
@common_options
@click.option("--tar-base", default="data", show_default=True, help="Base directory containing .tar shards.")
@click.option("--batch-size", default=32, show_default=True, type=int, help="Training batch size.")
def train_fpt(tar_base, batch_size, **common_kwargs):
    """Train the original FlowPokeTransformer (flow_poke.model)."""
    from flow_poke.model import FlowPokeTransformer_Base
    from flow_poke.data import TrackerShardsDataModule

    data = TrackerShardsDataModule(tar_base=tar_base, batch_size=batch_size, shuffle=1000)

    config_dict = dict(model="fpt", tar_base=tar_base, batch_size=batch_size, **common_kwargs)
    _train(
        data=data,
        model_cls=FlowPokeTransformer_Base,
        make_train_fns=fpt_make_train_fns,
        config_dict=config_dict,
        **common_kwargs,
    )


@cli.command("myriad")
@common_options
@click.option("--tar-base", required=True, multiple=True, help="Base directory/directories containing .tar shards.")
@click.option("--train-shards", default=None, help="Glob pattern for training shards (default: all *.tar).")
@click.option("--batch-size", default=8, show_default=True, type=int, help="Training batch size.")
@click.option("--num-workers", default=4, show_default=True, type=int, help="DataLoader worker count.")
@click.option("--num-tracks", default=16, show_default=True, type=int, help="Number of tracks per sample.")
@click.option("--num-steps", default=16, show_default=True, type=int, help="Number of time steps per sequence.")
@click.option("--shuffle", default=500, show_default=True, type=int, help="WebDataset shuffle buffer (0 = off).")
def train_2d(tar_base, train_shards, batch_size, num_workers, num_tracks, num_steps, shuffle, **common_kwargs):
    """Train on 2-D tracker shards (myriad.data_2d.TrackerShardsDataModule)."""
    from myriad.model import MyriadStepByStep_Large
    from myriad.data_2d import TrackerShardsDataModule

    data = TrackerShardsDataModule(
        tar_base=list(tar_base),
        batch_size=batch_size,
        num_workers=num_workers,
        num_tracks=num_tracks,
        num_steps=num_steps,
        train={"shards": train_shards, "shuffle": shuffle},
    )

    config_dict = dict(
        model="large", dataset="2d",
        tar_base=list(tar_base), batch_size=batch_size,
        num_workers=num_workers, num_tracks=num_tracks, num_steps=num_steps,
        train_shards=train_shards,
        **common_kwargs,
    )
    _train(
        data=data,
        model_cls=MyriadStepByStep_Large,
        make_train_fns=myriad_make_train_fns,
        config_dict=config_dict,
        **common_kwargs,
    )


@cli.command("billiards")
@common_options
@click.option("--batch-size", default=8, show_default=True, type=int, help="Training batch size.")
@click.option("--num-workers", default=8, show_default=True, type=int, help="DataLoader worker count.")
@click.option("--nr-balls", default=16, show_default=True, type=int, help="Number of balls in the simulation.")
@click.option("--frame-size", default=512, show_default=True, type=int, help="Rendered frame size in pixels.")
@click.option("--duration", default=0.5, show_default=True, type=float, help="Simulation duration in seconds.")
@click.option("--dt", default=0.01, show_default=True, type=float, help="Simulation time step.")
def train_billiards(batch_size, num_workers, nr_balls, frame_size, duration, dt, **common_kwargs):
    """Train on procedurally generated billiards data (myriad.data_billiards.BilliardSimDataModule)."""
    from myriad.model import MyriadStepByStep_Large_Billiard
    from myriad.data_billiards import BilliardSimDataModule

    data = BilliardSimDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        train={"dataset_config": dict(nr_balls=nr_balls, frame_size=frame_size, duration=duration, dt=dt)},
    )

    config_dict = dict(
        model="billiard", dataset="billiards",
        batch_size=batch_size, num_workers=num_workers,
        nr_balls=nr_balls, frame_size=frame_size, duration=duration, dt=dt,
        **common_kwargs,
    )
    _train(
        data=data,
        model_cls=MyriadStepByStep_Large_Billiard,
        make_train_fns=myriad_make_train_fns,
        config_dict=config_dict,
        **common_kwargs,
    )


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)

    try:
        cli()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
