import os
import csv
import torch
import hydra
import einops
import cv2
import time
import billiards
import click
import omegaconf
import types
import numpy as np

from tabulate import tabulate
from omegaconf import DictConfig
from pathlib import Path
from jaxtyping import Float, UInt8
from omegaconf import open_dict
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from jaxtyping import UInt8, Float
from mediapy import write_video

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from myriad.data_billiards import (
    simulate_billiard_game,
    render_billiard_frame,
    BilliardSimDataset,
    BilliardSimDataModule
)
from myriad.model import MyriadStepByStep

from .qual import seed_everything, get_model


# ---------------------------------------------------------------------------------------------------------------------
# Data Utilities
# ---------------------------------------------------------------------------------------------------------------------

def get_data(duration: int) -> tuple[torch.utils.data.DataLoader, BilliardSimDataset]:
    dataset_config = {
        "frame_size": 512,
        "border_offset_range": [0.05, 0.1],
        "ball_radius": 0.0333,
        "nr_balls": 16,
        "duration": duration,
        "p_moving": 0.0,
        "dt": 0.01,
    }
    cfg = {
        "batch_size": 1,
        "val_batch_size": 1,
        "num_workers": 8,
        "val_num_workers": 8,
        "train": {
            "dataset_config": dataset_config,
        },
        "validation": {
            "dataset_config": dataset_config,
        },
    }

    dmodule = BilliardSimDataModule(**cfg)
    loader = dmodule.val_dataloader()
    dataset = BilliardSimDataset(**dataset_config)
    return loader, dataset

def create_sample(
    loader: torch.utils.data.DataLoader,
    dataset: BilliardSimDataset,
    use_random_goal: bool,
    action_ball_idx: int
):
    val_batch = {k: v[:1] for k, v in next(iter(loader)).items()}

    num_traj = dataset.nr_balls
    c_pos = val_batch["pos_poke"]
    c_pos = einops.rearrange(c_pos, "1 (t n) c -> 1 t n c", n=num_traj)

    if use_random_goal:
        # use random goal that is within image borders
        border = val_batch["border_offsets"][0]
        goal_x = np.random.uniform(border[3] + dataset.ball_radius, dataset.frame_size - border[1] - dataset.ball_radius) / dataset.frame_size
        goal_y = np.random.uniform(border[0] + dataset.ball_radius, dataset.frame_size - border[2] - dataset.ball_radius) / dataset.frame_size
        goal_pos = (goal_x, goal_y)
    else:
        goal_pos = c_pos[0, -1, action_ball_idx].tolist()

    return val_batch, goal_pos

def setup_billiard_sim(
    batch: dict[str, torch.Tensor],
    action: torch.Tensor,
    dataset: BilliardSimDataset,
    b_idx: int,
    action_ball_idx: int
) -> billiards.Billiard:
    bounds = [
        billiards.InfiniteWall(
            (0, batch["border_offsets"][b_idx, 0]),
            (dataset.frame_size - batch["border_offsets"][b_idx, 1], batch["border_offsets"][b_idx, 0]),
        ),
        billiards.InfiniteWall(
            (dataset.frame_size - batch["border_offsets"][b_idx, 1], batch["border_offsets"][b_idx, 0]),
            (dataset.frame_size - batch["border_offsets"][b_idx, 1], dataset.frame_size - batch["border_offsets"][b_idx, 2]),
        ),
        billiards.InfiniteWall(
            (dataset.frame_size - batch["border_offsets"][b_idx, 1], dataset.frame_size - batch["border_offsets"][b_idx, 2]),
            (batch["border_offsets"][b_idx, 3], dataset.frame_size - batch["border_offsets"][b_idx, 2]),
        ),
        billiards.InfiniteWall(
            (batch["border_offsets"][b_idx, 3], dataset.frame_size - batch["border_offsets"][b_idx, 2]),
            (batch["border_offsets"][b_idx, 3], batch["border_offsets"][b_idx, 0]),
        ),
    ]

    bld = billiards.Billiard(obstacles=bounds)
    for ball_idx in range(dataset.nr_balls):
        bld.add_ball(
            pos=tuple((batch["pos"][b_idx, 0, ball_idx] * dataset.frame_size).tolist()),
            vel=(0.0, 0.0) if ball_idx != action_ball_idx else tuple((action[b_idx] * dataset.frame_size / dataset.dt).tolist()),  # only action ball gets initial velocity
            radius=dataset.ball_radius,
        )
    return bld



# ---------------------------------------------------------------------------------------------------------------------
# Predict Function(s)
# ---------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def _ours_pred_function(
    model: MyriadStepByStep,
    x: Float[torch.Tensor, "b h w c"],
    pos_orig: Float[torch.Tensor, "b n 2"],
    poke: Float[torch.Tensor, "b n 2"],
    dt: float,
    num_traj: int,
    n_sim_steps: int,
    **kwargs
):
    ts = torch.arange(0, n_sim_steps, device=x.device, dtype=x.dtype).unsqueeze(0)
    ts = ts.repeat(x.size(0), 1)
    
    pos0 = pos_orig.clone()
    pos1 = pos0 + poke
    given_pos = torch.stack([pos0, pos1], dim=1)
    given_pos = einops.rearrange(given_pos, "b t n c -> b (t n) c")

    camera_static = torch.ones((x.size(0),), device=x.device, dtype=torch.bool)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        d_img = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in model.embed_image(x).items()}
        pred_sim = model.predict_simulate(
            n_traj=num_traj,
            ts=ts,
            given_pos=given_pos,
            camera_static=camera_static,
            d_img=d_img,
            verbose=False,
        )

    return pred_sim

# ---------------------------------------------------------------------------------------------------------------------
# Qualitative Visualization Functions
# ---------------------------------------------------------------------------------------------------------------------

def visualize_state(
    dataset: BilliardSimDataset,
    pos: Float[torch.Tensor, "... n c"],
    border_offsets: Float[torch.Tensor, "4"],
    goal_pos: tuple[float, float] | None = None
) -> UInt8[torch.Tensor, "t h w c"]:
    *d_pos, _, _ = pos.shape
    pos = einops.rearrange(pos, "... n c -> (...) n c")
    frames = []
    for i in range(pos.size(0)):
        frames.append(
            einops.rearrange(
                torch.from_numpy(
                    render_billiard_frame(
                        (pos[i] * dataset.frame_size).double().numpy(),
                        ball_rad=[dataset.ball_radius] * pos.size(1),
                        frame_size=dataset.frame_size,
                        border_offsets=border_offsets.numpy().tolist(),
                        antialiasing=True,
                        goal_pos=(goal_pos[0] * dataset.frame_size).float().cpu().numpy() if goal_pos is not None else None,
                    )
                ) / 127.5 - 1, "h w c -> c h w",
            )
        )
    return torch.stack(frames).view(*d_pos, *frames[-1].shape)

def save_qualitative(
    sample_dir: str,
    dataset: BilliardSimDataset,
    sim: Float[torch.Tensor, "t n c"],
    batch: dict[str, torch.Tensor],
    goal_pos: tuple[float, float],
    action_ball_idx: int,
    best_action: torch.Tensor,
    best_idx: int,
):
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # get video with predicted trajectory
    pred_video = visualize_state(dataset, sim[0].cpu(), batch["border_offsets"][0])
    pred_video = pred_video[:best_idx+1]

    # get rollout with best action
    batch_planned = get_batch_with_best_action(batch, best_action, dataset, action_ball_idx)

    bld = setup_billiard_sim(
        batch_planned,
        best_action.unsqueeze(0) if best_action.ndim == 1 else best_action,
        dataset,
        b_idx=0,
        action_ball_idx=action_ball_idx
    )

    _, pos, _, _ = simulate_billiard_game(
        bld=bld,
        duration=(best_idx+1) * dataset.dt,
        dt=dataset.dt,
    )
    pos = torch.from_numpy(np.array(pos)).float()
    print(f"{pos.shape=}, {sim.shape=}, {goal_pos=}")   # pos.shape=torch.Size([3, 16, 2]), sim.shape=torch.Size([1, 25, 16, 2]), goal_pos=tensor([[0.5402, 0.2222]], device='cuda:0')
    # `simulate_billiard_game` returns positions in pixel coordinates; normalize for `visualize_state`.
    pos_norm = pos / dataset.frame_size
    rollout_video = visualize_state(dataset, pos_norm.unsqueeze(0), batch["border_offsets"][0])[0]
    print(f"{rollout_video.shape=}, {pred_video.shape=}, {best_idx=}")  # rollout_video.shape=torch.Size([3, 3, 512, 512]), pred_video.shape=torch.Size([3, 3, 512, 512])

    # plot setting
    start_frame = pred_video[0]
    start_frame = ((start_frame + 1) * 127.5).round().to(torch.uint8)
    start_frame = einops.rearrange(start_frame, "c h w -> h w c").numpy()

    goal_pos = goal_pos.squeeze().float().cpu().numpy()
    goal_pos[0] = goal_pos[0] * dataset.frame_size
    goal_pos[1] = goal_pos[1] * dataset.frame_size

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(start_frame)
    ax.scatter(goal_pos[0], goal_pos[1], color="blue", s=100, label="Goal", marker="x", linewidths=5.0)
    ax.axis("off")
    fig.savefig(os.path.join(sample_dir, "start_setting.pdf"), bbox_inches="tight", pad_inches=0)
    fig.savefig(os.path.join(sample_dir, "start_setting.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # GT rollout plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(start_frame)

    for n in range(pos.shape[1]):
        c_traj = pos[:, n]
        ax.plot(
            c_traj[:, 0], c_traj[:, 1],
            color="green",
            linewidth=8.0,
            label="Rollout with Best Action" if n == 0 else None,
        )
    ax.scatter(goal_pos[0], goal_pos[1], color="blue", s=100, label="Goal", marker="x", linewidths=5.0)
    ax.axis("off")
    fig.savefig(os.path.join(sample_dir, "sim_rollout.pdf"), bbox_inches="tight", pad_inches=0)
    fig.savefig(os.path.join(sample_dir, "sim_rollout.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Pred rollout plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(start_frame)

    for n in range(sim.shape[2]):
        c_traj = sim[0, :best_idx+1, n] * dataset.frame_size
        ax.plot(
            c_traj[:, 0], c_traj[:, 1],
            color="orange",
            linewidth=8.0,
            label="Predicted Trajectory" if n == 0 else None,
        )
    ax.scatter(goal_pos[0], goal_pos[1], color="blue", s=100, label="Goal", marker="x", linewidths=5.0)
    ax.axis("off")
    fig.savefig(os.path.join(sample_dir, "model_rollout.pdf"), bbox_inches="tight", pad_inches=0)
    fig.savefig(os.path.join(sample_dir, "model_rollout.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # GT individual frames
    gt_frames_dir = sample_dir / "sim_frames"
    gt_frames_dir.mkdir(parents=True, exist_ok=True)
    sim_frames = []
    for t in range(best_idx + 1):
        c_frame = rollout_video[t].float().cpu().numpy()
        c_frame = ((c_frame + 1) * 127.5).round().astype(np.uint8)
        c_frame = einops.rearrange(c_frame, "c h w -> h w c")

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(c_frame)
        ax.axis("off")
        ax.scatter(goal_pos[0], goal_pos[1], color="blue", s=100, label="Goal", marker="x", linewidths=5.0)
        fig.savefig(gt_frames_dir / f"frame_{t:04d}.png", bbox_inches="tight", pad_inches=0)
        fig.canvas.draw()  # Ensure the canvas is drawn before converting to numpy
        sim_frames.append(np.array(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    write_video(str(sample_dir / "sim_rollout.mp4"), sim_frames, fps=30)

    # Pred individual frames
    pred_frames_dir = sample_dir / "pred_frames"
    pred_frames_dir.mkdir(parents=True, exist_ok=True)
    pred_frames = []
    for t in range(best_idx + 1):
        c_frame = pred_video[t].float().cpu().numpy()
        c_frame = ((c_frame + 1) * 127.5).round().astype(np.uint8)
        c_frame = einops.rearrange(c_frame, "c h w -> h w c")

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(c_frame)
        ax.axis("off")
        ax.scatter(goal_pos[0], goal_pos[1], color="blue", s=100, label="Goal", marker="x", linewidths=5.0)
        fig.savefig(pred_frames_dir / f"frame_{t:04d}.png", bbox_inches="tight", pad_inches=0)
        fig.canvas.draw()  # Ensure the canvas is drawn before converting to numpy
        pred_frames.append(np.array(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    write_video(str(sample_dir / "model_rollout.mp4"), pred_frames, fps=30)

    print(f"Saved qualitative results to {sample_dir}")


# ---------------------------------------------------------------------------------------------------------------------
# Planning Functions
# ---------------------------------------------------------------------------------------------------------------------

def get_batch_with_best_action(
    batch: dict[str, torch.Tensor],
    best_action: torch.Tensor,
    dataset: BilliardSimDataset,
    action_ball_idx: int
):
    # replace original flow data with updated flow that corresponds with action
    batch_planned = deepcopy(batch)
    new_flow = batch_planned["flow_poke"]
    new_flow = einops.rearrange(new_flow, "b (t n_traj) c -> b t n_traj c", n_traj=dataset.nr_balls)
    new_flow[:, 0] = torch.zeros_like(new_flow[:, 0])
    new_flow[:, 0, action_ball_idx] = best_action.to(device="cuda:0", dtype=torch.float32)
    new_flow = einops.rearrange(new_flow, "b t n_traj c -> b (t n_traj) c")
    batch_planned["flow_poke"] = new_flow
    batch_planned["flow_query"] = new_flow

    pos = batch_planned["pos_poke"].to(device="cuda:0", dtype=torch.float32)
    pos = einops.rearrange(pos, "b (t n_traj) c -> b t n_traj c", n_traj=dataset.nr_balls)
    batch_planned["pos"] = pos
    return batch_planned

def test_rollouts(
    num_test_rollouts: int,
    batch: dict[str, torch.Tensor],
    best_action: torch.Tensor,
    n_sim_steps: int,
    action_ball_idx: int,
    goal_ball_idx: int,
    goal_pos: tuple[float, float],
    success_thresh: float,
    dataset: BilliardSimDataset,
    best_sim: torch.Tensor,
):
    distances = []

    for _ in range(num_test_rollouts):
        batch_planned = get_batch_with_best_action(batch, best_action, dataset, action_ball_idx)

        bld = setup_billiard_sim(
            batch_planned,
            best_action.unsqueeze(0) if best_action.ndim == 1 else best_action,
            dataset,
            b_idx=0,
            action_ball_idx=action_ball_idx
        )

        ts, pos, vel, col = simulate_billiard_game(
            bld=bld,
            duration=n_sim_steps * dataset.dt,
            dt=dataset.dt,
        )

        distances_to_goal = np.linalg.norm(np.array(pos)[:, goal_ball_idx] - np.array(goal_pos) * dataset.frame_size, axis=-1)
        min_idx, min_dist = np.argmin(distances_to_goal), np.min(distances_to_goal)
        distances.append(min_dist)

    dists = np.array(distances)
    mean_min_dist = np.mean(dists)
    success_rate = np.mean(dists < success_thresh * dataset.frame_size)

    print(f"Empirical mean min distance: {dists.mean():.1f}px ± {dists.std(ddof=1):.1f}px")
    print(f"Success rate: {success_rate:.1%}")

    return dists, mean_min_dist, success_rate

@torch.no_grad()
def plan_action_ensembled(
    model: MyriadStepByStep,
    pred_function: types.FunctionType,
    batch: dict[str, torch.Tensor],
    goal_pos: tuple[float, float],
    action_ball_idx: int,
    goal_ball_idx: int,
    num_traj: int,
    batch_size: int,
    time_limit: float,
    max_action_mag: float,
    sim_steps: int,
    ensemble_size: int,
    success_thresh: float,
    dataset: BilliardSimDataset,
):
    assert batch["flow_poke"].shape[0] == 1, "Planning only implemented for single batch element"
    assert ensemble_size >= 1, "Ensemble size must be greater than 1"
    assert time_limit > 0, "Time limit must be positive"
    assert batch_size > 0, "Batch size must be positive"
    
    # prepare data
    goal_pos = torch.as_tensor(goal_pos, device="cuda:0", dtype=torch.float32)
    x_single = batch["x"].to(device="cuda:0", dtype=torch.float32)
    pos_poke = batch["pos_poke"].to(device="cuda:0", dtype=torch.float32)
    pos_single = einops.rearrange(pos_poke, "b (t n) c -> b t n c", n=num_traj)[:, 0]
    flow = einops.rearrange(batch["flow_poke"].to(device="cuda:0", dtype=torch.float32), "b (t n) c -> b t n c", n=num_traj)
    flow_shape = flow.shape
    t_poke = einops.rearrange(batch["t_poke"].to(device="cuda:0", dtype=torch.float32), "b (t n_traj) -> b t n_traj", n_traj=num_traj)
    dt = dataset.dt

    # intialize planning loop
    best_dist, best_action, best_idx, best_sim = float('inf'), None, None, None

    start_t = time.perf_counter()
    deadline = start_t + time_limit * 60
    num_actions = 0

    # planning loop
    with tqdm(total=None, desc="Simulating actions", dynamic_ncols=True, unit=" actions") as pbar:
        while True:
            # get batch_size random actions with 1) random direction and 2) random magnitude up to max_action_mag
            unnorm = torch.randn(batch_size, 2, device="cuda:0", dtype=torch.float32)
            norms = unnorm.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            action_mag = torch.rand(batch_size, 1, device="cuda:0", dtype=torch.float32) * max_action_mag
            batch_actions = ((unnorm / norms) * action_mag).to(torch.float32)

            # get batched data
            x_batched = x_single.expand(batch_size * ensemble_size, *x_single.shape[1:])
            pos_batched = pos_single.expand(batch_size * ensemble_size, *pos_single.shape[1:])
            action_poke = torch.zeros((batch_size * ensemble_size, *flow_shape[2:]), device="cuda:0", dtype=torch.float32)
            actions_rep = batch_actions.repeat_interleave(ensemble_size, dim=0)
            action_poke[:, action_ball_idx] = actions_rep

            # predict simulation
            pred_sim = pred_function(
                model=model,
                x=x_batched,
                pos_orig=pos_batched,
                poke=action_poke,
                dt=dt,
                num_traj=num_traj,
                n_sim_steps=sim_steps,
                action_idx=action_ball_idx,
                border_offsets=batch["border_offsets"],
                dataset=dataset
            )

            torch.cuda.synchronize()

            if time.perf_counter() >= deadline and num_actions > 0:  # ensure we get at least one batch per model
                break

            num_actions += batch_size

            # test how close we got
            traj = pred_sim[:, :, goal_ball_idx]
            traj = traj.view(batch_size, ensemble_size, traj.size(1), traj.size(2))
            goal_pos_b = goal_pos.view(1, 1, 1, -1)
            dists = (traj - goal_pos_b).norm(dim=-1)
            min_dists, min_idxs = torch.min(dists, dim=2)
            mean_min_dists = min_dists.mean(dim=1)  # (batch_size,)

            # update best
            batch_best_val, batch_best_i = torch.min(mean_min_dists, dim=0)
            candidate_dist = float(batch_best_val.item())
            if candidate_dist < best_dist:
                i = int(batch_best_i.item())
                best_dist = candidate_dist
                best_action = batch_actions[i].detach().cpu()
                first_idx = i * ensemble_size

                mean_dists_over_time = dists[i].mean(dim=0)
                best_idx = int(mean_dists_over_time.argmin().item())

                # keep only the winning element from pred_sim
                best_sim = pred_sim[first_idx].detach().cpu()
                pbar.update(batch_size)
            
            # break if time is up or task is solved
            if time.perf_counter() >= deadline and num_actions > 0:  # don't start new prediction if time is up
                break
            if best_dist < success_thresh:
                print("Breaking because solved!", flush=True)
                break
    
    if num_actions == 0:
        raise ValueError("No actions were evaluated.")
    if best_action is None or best_sim is None or best_idx is None:
        raise ValueError("No valid action found within the time limit.")
    
    return best_action.float(), best_dist, best_sim, best_idx, num_actions

def test_one_example(
    model: MyriadStepByStep,
    loader: torch.utils.data.DataLoader,
    dataset: BilliardSimDataset,
    pred_function: types.FunctionType,
    use_random_goal: bool,
    action_ball_idx: int,
    goal_ball_idx: int,
    num_test_rollouts: int,
    batch_size: int,
    time_limit_min: float,
    max_action_mag: float,
    sim_steps: int,
    ensemble_size: int,
    success_thresh: float,
    sample_dir: str | None = None,
):
    batch, goal_pos = create_sample(loader, dataset, use_random_goal, action_ball_idx)

    best_action, _, best_sim, best_idx, _ = plan_action_ensembled(
        model=model,
        pred_function=pred_function,
        batch=batch,
        goal_pos=goal_pos,
        action_ball_idx=action_ball_idx,
        goal_ball_idx=goal_ball_idx,
        num_traj=dataset.nr_balls,
        batch_size=batch_size,
        time_limit=time_limit_min,
        max_action_mag=max_action_mag,
        sim_steps=sim_steps,
        ensemble_size=ensemble_size,
        success_thresh=success_thresh,
        dataset=dataset,
    )

    if sample_dir is not None:
        os.makedirs(sample_dir, exist_ok=True)
        goal_pos_t = torch.as_tensor(goal_pos, device="cuda:0", dtype=torch.float32).unsqueeze(0)
        save_qualitative(sample_dir, dataset, best_sim.unsqueeze(0), batch, goal_pos_t, action_ball_idx, best_action, best_idx)

    _, mean_min_dist, success_rate = test_rollouts(
        num_test_rollouts=num_test_rollouts,
        batch=batch,
        best_action=best_action,
        n_sim_steps=sim_steps,
        action_ball_idx=action_ball_idx,
        goal_ball_idx=goal_ball_idx,
        goal_pos=goal_pos,
        success_thresh=success_thresh,
        dataset=dataset,
        best_sim=best_sim
    )

    return success_rate, mean_min_dist

# ---------------------------------------------------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------------------------------------------------

@click.command()
@click.option("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file containing the model weights and config.")
@click.option("--duration", type=float, default=1.0, help="Duration of the billiard simulation in seconds.")
@click.option("--method", type=str, default="ours", help="Planning method to use (default: 'ours').")
@click.option("--out_dir", type=str, default="eval_results/billiard_planning", help="Directory to save evaluation results.")
@click.option("--tag", type=str, default=None, help="Optional tag to distinguish this evaluation run.")
@click.option("--num_warmup", type=int, default=5, help="Number of warmup runs to perform before actual evaluation.")
@click.option("--use_random_goal", is_flag=True, help="Whether to use a random goal position within the image borders instead of the final position in the trajectory.")
@click.option("--action_ball_idx", type=int, default=0, help="Index of the ball to apply the action to.")
@click.option("--goal_ball_idx", type=int, default=0, help="Index of the ball to evaluate success on (i.e., the ball that should reach the goal).")
@click.option("--num_samples", type=int, default=50, help="Number of random samples to evaluate.")
@click.option("--num_test_rollouts", type=int, default=10, help="Number of test rollouts to perform for each planned action to empirically evaluate success rate.")
@click.option("--batch_size", type=int, default=512, help="Number of actions to evaluate in each batch during planning.")
@click.option("--time_limit_min", type=float, default=15, help="Time limit for the planning process in minutes (per sample).")
@click.option("--max_action_mag", type=float, default=0.01, help="Maximum magnitude of the poke action (as a fraction of the frame size).")
@click.option("--ensemble_size", type=int, default=1, help="Number of ensemble members to use when evaluating each action during planning.")
@click.option("--success_thresh", type=float, default=0.0333, help="Distance threshold (as a fraction of the frame size) for considering a rollout successful.")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
def main(
    checkpoint_path: str,
    duration: float = 1.0,
    method: str = "ours",
    out_dir: str = "eval_results/billiard_planning",
    tag: str | None = None,
    num_warmup: int = 5, 
    use_random_goal: bool = True,
    action_ball_idx: int = 0,
    goal_ball_idx: int = 0,
    num_samples: int = 50,
    num_test_rollouts: int = 10,
    batch_size: int = 512,
    time_limit_min: float = 15,
    max_action_mag: float = 0.01,
    ensemble_size: int = 1,
    success_thresh: float = 0.0333,
    seed: int=42,
):
    seed_everything(seed)

    out_path = Path(out_dir)
    if tag is not None:
        out_path = out_path / tag
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_path / now
    out_path.mkdir(parents=True, exist_ok=True)

    pred_function = None
    if method == "ours":
        model = get_model("billiard", checkpoint_path)
        pred_function = _ours_pred_function
    else:
        raise NotImplementedError(f"Method {method} not implemented.")

    model = model.to("cuda:0")

    loader, dataset = get_data(duration)

    # Warmup runs
    for i_warmup in range(num_warmup):
        print(f"Warmup run {i_warmup+1}/{num_warmup}...")
        test_one_example(
            model=model,
            loader=loader,
            dataset=dataset,
            pred_function=pred_function,
            use_random_goal=use_random_goal,
            action_ball_idx=action_ball_idx,
            goal_ball_idx=goal_ball_idx,
            num_test_rollouts=num_test_rollouts,
            batch_size=batch_size,
            time_limit_min=1,
            max_action_mag=max_action_mag,
            sim_steps=round(duration / dataset.dt),
            ensemble_size=ensemble_size,
            success_thresh=success_thresh,
        )

    torch.cuda.synchronize()

    # Actual planning runs with timelimit enforced
    all_results = []
    for i_run in range(num_samples):
        print(f"\n#######\nSample {i_run+1}/{num_samples}\n#######")
        sample_dir = os.path.join(out_path, f"sample_{i_run:04d}") 
        all_results.append(
            test_one_example(
                model=model,
                loader=loader,
                dataset=dataset,
                pred_function=pred_function,
                use_random_goal=use_random_goal,
                action_ball_idx=action_ball_idx,
                goal_ball_idx=goal_ball_idx,
                num_test_rollouts=num_test_rollouts,
                batch_size=batch_size,
                time_limit_min=time_limit_min,
                max_action_mag=max_action_mag,
                sim_steps=round(duration / dataset.dt),
                ensemble_size=ensemble_size,
                success_thresh=success_thresh,
                sample_dir=sample_dir,
            )
        )
    
    # Save results
    results_path = os.path.join(out_path, "all_results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["success_rate", "mean_min_dist"])
        writer.writerows(all_results)

    # save summary results
    results_arr = np.asarray(all_results, dtype=float)
    if results_arr.size:
        success_rates = results_arr[:, 0]
        mean_min_dists = results_arr[:, 1]

        summary_rows = [
            ("mean_success_rate", float(np.mean(success_rates))),
            ("mean_mean_min_dist", float(np.mean(mean_min_dists))),
            ("std_mean_min_dist", float(np.std(mean_min_dists))),
        ]

        summary_table = tabulate(summary_rows, headers=["metric", "value"], tablefmt="github", floatfmt=".6f")
        print(summary_table)

        summary_path = os.path.join(out_path, "summary.txt")
        with open(summary_path, "w", newline="") as f:
            f.write(summary_table)

    print("Finished billiard planning eval successfully! 🥳")


if __name__ == "__main__":
    # TF32
    if torch.__version__.startswith("2.9"):
        torch.backends.fp32_precision = "tf32"
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.fp32_precision = "tf32"
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Enable larger number of compilation shapes
    torch._dynamo.config.cache_size_limit = max(2**15, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_recompile_limit = max(2**15, torch._dynamo.config.accumulated_recompile_limit)

    from einops._torch_specific import allow_ops_in_compiled_graph

    allow_ops_in_compiled_graph()

    main()
