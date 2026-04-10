import random
import click
import types
import torch
import hydra
import einops

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from datetime import datetime
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from functools import partial

from myriad.model import MyriadStepByStep, RFHeadDistribution, FusedTransformerLayer, MyriadStepByStep_Large, MyriadStepByStep_Large_Billiard


# ---------------------------------------------------------------------------------------------------------------------
# General Utilities
# ---------------------------------------------------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------------------------------------------------
# Example Data
# ---------------------------------------------------------------------------------------------------------------------

class Example:

    def __init__(self, img_path: str, background_queries: list, object_queries: list, object_pokes: list):
        self.img_path = img_path
        self.background_queries = background_queries
        self.background_pokes = [(0.0, 0.0)]*len(background_queries)
        self.object_queries = object_queries
        self.object_pokes = object_pokes
    
    def get_image(self):
        return Image.open(self.img_path).convert("RGB")
    
    def get_queries(self) -> list[tuple[float, float]]:
        return self.background_queries + self.object_queries

    def get_pokes(self, mode='fully_poked') -> list[tuple[float, float]] | None:
        if mode == 'fully_poked':
            return self.background_pokes + self.object_pokes
        if mode == "first_query":
            return self.background_pokes + self.object_pokes[:1]
        elif mode == 'unpoked':
            return None
        elif mode == 'background_fixed':
            return self.background_pokes + [(np.nan, np.nan)]*len(self.object_queries)
        elif mode == 'object_coherence':
            return self.background_pokes + self.object_pokes[:1] + [(np.nan, np.nan)]*len(self.object_pokes[1:])
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")
        
    def get_pokes_queries(self, mode='fully_poked'):
        if mode == 'fully_poked':
            return self.background_pokes + self.object_pokes, self.get_queries()
        if mode == "first_query":
            return self.background_pokes + self.object_pokes[:1], self.background_queries + self.object_queries[:1]
        elif mode == 'unpoked':
            return None, self.get_queries()
        elif mode == 'background_fixed':
            return self.background_pokes + [(np.nan, np.nan)]*len(self.object_queries), self.get_queries()
        elif mode == 'object_coherence':
            return self.background_pokes + self.object_pokes[:1] + [(np.nan, np.nan)]*len(self.object_pokes[1:]), self.get_queries()
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

EXAMPLES = {
    "falling_ball": Example(
        img_path='eval/qual_examples/falling_ball.jpg',
        background_queries=[(0.2, 0.2), (0.8, 0.8), (0.7, 0.7), (0.1, 0.7), (0.08, 0.75), (0.08, 0.8), (0.9, 0.1), (0.5, 0.1), (0.2, 0.1)],
        object_queries=[(0.31, 0.61), (0.28, 0.60), (0.33, 0.64), (0.34, 0.57), (0.28, 0.68)],
        object_pokes=[(-0.02, 0.05), (-0.02, 0.05), (-0.02, 0.05), (-0.02, 0.05), (-0.02, 0.05)]
    ),

    "falling_person": Example(
        img_path='eval/qual_examples/falling_person.jpg',
        background_queries=[(0.31, 0.61), (0.2, 0.2), (0.8, 0.9), (0.7, 0.8), (0.1, 0.7), (0.08, 0.85), (0.08, 0.8), (0.9, 0.1), (0.5, 0.1), (0.2, 0.1)],
        object_queries=[(0.49, 0.32), (0.48, 0.28), (0.48, 0.37), (0.51, 0.31)],
        object_pokes=[(0.005, 0.025), (0.005, 0.025), (0.005, 0.025), (0.005, 0.025)],
    ),

    "car_speeding": Example(
        img_path='eval/qual_examples/car_speeding.jpg',
        background_queries=[(0.31, 0.45), (0.158, 0.2), (0.85, 0.9), (0.7, 0.95), (0.05, 0.5), (0.08, 0.85), (0.9, 0.1), (0.5, 0.1), (0.75, 0.4)],
        object_queries=[(0.3, 0.62)],
        object_pokes=[(0.015, 0.005)]
    ),

    "train": Example(
        img_path='eval/qual_examples/train_curved_track.jpg',
        background_queries=[(0.1, 0.1), (0.9, 0.95), (0.8, 0.95), (0.9, 0.75), (0.9, 0.1), (0.05, 0.5), (0.1, 0.8), (0.2, 0.9), (0.4, 0.9), (0.5, 0.05)],
        object_queries=[(0.64, 0.5), (0.67, 0.74), (0.6, 0.63), (0.68, 0.57)],
        object_pokes=[(0.02, 0.01), (0.02, 0.01), (0.02, 0.01), (0.02, 0.01)]
    ),

    "ball_roll": Example(
        img_path='eval/qual_examples/ball_roll.jpg',
        background_queries=[(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9), (0.6, 0.25), (0.25, 0.5), (0.75, 0.4), (0.55, 0.75), (0.95, 0.6)],
        object_queries=[(0.27, 0.72)],
        object_pokes=[(0.05, 0.0)],
    ),

    "egg_roll": Example(
        img_path='eval/qual_examples/egg_roll.jpg',
        background_queries=[(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9), (0.6, 0.25), (0.25, 0.5), (0.75, 0.4), (0.55, 0.75), (0.95, 0.6)],
        object_queries=[(0.27, 0.72)],
        object_pokes=[(0.05, 0.0)],
    ),

    "floor_jump": Example(
        img_path="eval/qual_examples/floor_jump.png",
        background_queries=[(0.1, 0.1), (0.9, 0.1), (0.2, 0.9), (0.8, 0.9), (0.5, 0.5), (0.15, 0.3), (0.7, 0.4), (0.25, 0.6), (0.8, 0.75), (0.6, 0.3)],
        object_queries=[(0.45, 0.28)],
        object_pokes=[(0.0, 0.05)],
    ),

    "trampoline_jump": Example(
        img_path="eval/qual_examples/trampoline_jump.jpg",
        background_queries=[(0.1, 0.1), (0.9, 0.1), (0.2, 0.9), (0.8, 0.9), (0.5, 0.5), (0.15, 0.3), (0.7, 0.4), (0.25, 0.6), (0.8, 0.75), (0.6, 0.3)],
        object_queries=[(0.45, 0.28)],
        object_pokes=[(0.0, 0.05)],
    ),

    "elephant_walk": Example(
        img_path="eval/qual_examples/elephant_walk.png",
        background_queries=[(0.1, 0.1), (0.75, 0.2), (0.25, 0.38), (0.85, 0.4), (0.65, 0.5), (0.75, 0.6), (0.7, 0.85), (0.35, 0.95), (0.9, 0.9), (0.05, 0.78)],
        object_queries=[(0.33, 0.645), (0.25, 0.56), (0.47, 0.55)],
        object_pokes=[(0.03, 0.0), (0.03, 0.0), (0.03, 0.0)],
    ),

    "people_walking": Example(
        img_path="eval/qual_examples/people_walking.png",
        background_queries = [(0.1, 0.1), (0.75, 0.2), (0.25, 0.38), (0.85, 0.6), (0.7, 0.5), (0.8, 0.6), (0.8, 0.85), (0.35, 0.95), (0.9, 0.9), (0.05, 0.78)],
        object_queries=[(0.4, 0.4), (0.6, 0.35), (0.28, 0.75)],
        object_pokes=[(0.03, 0.008), (0.03, 0.008), (0.02, 0.015)],
    ),

    "jenga": Example(
        img_path="eval/qual_examples/jenga_pull.png",
        background_queries=[(0.1, 0.1), (0.9, 0.05), (0.02, 0.9), (0.2, 0.85), (0.8, 0.9), (0.99, 0.89)],
        object_queries=[(0.8, 0.52), (0.35, 0.47), (0.23, 0.27), (0.7, 0.07)],
        object_pokes=[(0.04, -0.015), (1e-7, 1e-7), (1e-7, 1e-7), (1e-7, 1e-7)]
    ),

    "clouds": Example(
        img_path="eval/qual_examples/clouds.png",
        background_queries = [(0.1, 0.1), (0.95, 0.95), (0.95, 0.05), (0.45, 0.8), (0.2, 0.75), (0.7, 0.72), (0.05, 0.7), (0.55, 0.02), (0.02, 0.97)],
        object_queries = [(0.25, 0.25), (0.78, 0.4), (0.63, 0.15), (0.3, 0.45)],
        object_pokes = [(0.03, -0.005), (0.025, -0.003), (0.028, -0.0045), (0.028, -0.0045)],
    ),

    "horse_rider": Example(
        img_path="eval/qual_examples/horse_rider.png",
        background_queries=[(0.1, 0.1), (0.9, 0.05), (0.05, 0.95), (0.95, 0.97), (0.45, 0.97), (0.15, 0.7)],
        object_queries = [(0.5, 0.5), (0.3, 0.4), (0.58, 0.25), (0.75, 0.52)],
        object_pokes = [(-0.03, 0.0), (-0.03, 0.0), (-0.03, 0.0), (-0.03, 0.0)]
    ),

    "weight_lifting": Example(
        img_path="eval/qual_examples/weight_lifting.png",
        background_queries = [(0.9, 0.05), (0.99, 0.8), (0.8, 0.97), (0.1, 0.85)],
        object_queries = [(0.64, 0.45), (0.35, 0.3), (0.9, 0.4)],
        object_pokes = [(0.008, -0.04), (0.004, -0.04), (0.003, -0.04)]
    ),

    "car_crossing": Example(
        img_path="eval/qual_examples/car_crossing.png",
        background_queries = [(0.9, 0.05), (0.99, 0.8), (0.8, 0.97), (0.1, 0.85)],
        object_queries = [(0.78, 0.81)],
        object_pokes = [(-0.01, -0.03)]
    ),

    "soccer": Example(
        img_path="eval/qual_examples/soccer.png",
        background_queries = [(0.9, 0.05), (0.99, 0.8), (0.8, 0.97), (0.1, 0.85)],
        object_queries = [(0.49, 0.67), (0.71, 0.5), (0.71, 0.44)],
        object_pokes = [(-0.01, -0.05), (-0.03, -0.02), (-0.03, -0.02)],
    )
}

def get_example(example_name: str, mode: str):
    example = EXAMPLES[example_name]
    img = example.get_image()
    pokes, queries = example.get_pokes_queries(mode=mode)

    num_trajectories = len(queries)
    if pokes is None:
        pts = torch.as_tensor(queries).unsqueeze(0)
    else:
        pos1 = torch.as_tensor(queries, dtype=torch.float32)
        flow = torch.as_tensor(pokes, dtype=torch.float32)    # [N, C]

        valid = ~torch.isnan(flow).any(dim=-1)
        valid = valid & torch.isfinite(pos1).all(dim=-1)

        idx_known = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        idx_unk   = torch.nonzero(~valid, as_tuple=False).squeeze(-1)
        perm = torch.cat([idx_known, idx_unk], dim=0) if idx_unk.numel() > 0 else idx_known
        K = int(valid.sum().item()) 

        # ensure known flows form a continuous prefix
        pos1p = pos1[perm] 
        flowp = flow[perm]

        pos_t0 = pos1p.unsqueeze(0)
        pos_t1_known = (pos1p[:K] + flowp[:K]).unsqueeze(0)  # cut to known only
        pts = torch.cat([pos_t0, pos_t1_known], dim=1)
    return img, pts, num_trajectories

def get_batch(img: Image.Image, pts: torch.Tensor, num_traj: int, num_steps: int) -> dict[str, torch.Tensor]:
    x = TF.to_tensor(img).unsqueeze(0).to('cuda:0')
    x = ((x * 2.0) - 1.0) # scale to [-1, 1]

    t_poke = torch.linspace(0, (num_steps - 1) / 12, num_steps) * 30
    t_poke = einops.repeat(t_poke, "t -> b t n_traj", b=pts.shape[0], n_traj=num_traj).to('cuda:0')

    return {
        'x': x,
        't_poke': einops.rearrange(t_poke, "b t n_traj -> b (t n_traj)"),
        'camera_static': torch.ones((pts.shape[0],), dtype=torch.bool).to('cuda:0'),
        'pos_poke': pts.to('cuda:0'),  # [b, N+K, 2]: t=0 for all N, then t=1 for K known
    }

# ---------------------------------------------------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------------------------------------------------

def _compile_cudagraph_wrap(fn, mode: str="default"):
    compiled_fn = torch.compile(fn, dynamic=False, fullgraph=True, mode=mode)

    def wrapped_fn(*args, **kwargs):
        res = compiled_fn(*args, **kwargs)
        if isinstance(res, torch.Tensor):
            return res.clone()
        elif isinstance(res, tuple):
            return tuple(r.clone() if isinstance(r, torch.Tensor) else r for r in res)
        else:
            raise NotImplementedError(f"Unsupported return type: {type(res)}")

    return wrapped_fn

def get_model(model_id: str, checkpoint_path: str, compile_mode: str = "default", compile_full_head_sampling_loop: bool = False) -> tuple[MyriadStepByStep, DictConfig]:
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.is_file(), f"Checkpoint path {checkpoint_path} does not exist or is not a file."

    model = None
    if model_id == "openset":
        model = MyriadStepByStep_Large()
    elif model_id == "billiard":
        model = MyriadStepByStep_Large_Billiard()
    else:
        raise ValueError(f"Unknown model_id {model_id}")

    model.load_state_dict(torch.load(checkpoint_path, weights_only=False, map_location="cpu")["model"], strict=True)

    model.eval()
    model.to("cuda:0")
    model.requires_grad_(False)

    if isinstance(model.distribution_head, RFHeadDistribution):
        model.distribution_head: RFHeadDistribution
        fwd = model.distribution_head.forward
        if hasattr(fwd, "_torchdynamo_orig_callable"):
            model.distribution_head.forward = partial(
                fwd._torchdynamo_orig_callable,
                self=model.distribution_head,
            )
        elif isinstance(fwd, types.MethodType):
            # already a bound method, keep as-is
            model.distribution_head.forward = fwd
        else:
            # plain function assigned on the instance, bind it manually
            model.distribution_head.forward = partial(fwd, self=model.distribution_head)
        
        if not compile_full_head_sampling_loop:
            model.distribution_head.head.forward_cached = torch.compile(
                model.distribution_head.head.forward_cached, dynamic=False, fullgraph=True, mode=compile_mode
            )
        else:
            model.distribution_head.head.sample_inner = torch.compile(
                model.distribution_head.head.sample_inner, dynamic=False, fullgraph=True, mode=compile_mode
            )
    
    for layer in model.transformer.mid_level:
        layer: FusedTransformerLayer
        layer._fwd_1 = torch.compile(layer._fwd_1, dynamic=False, fullgraph=True, mode=compile_mode)
    model.embed_image = torch.compile(model.embed_image, dynamic=False, fullgraph=True, mode=compile_mode)
    model.transformer._fwd_1 = _compile_cudagraph_wrap(model.transformer._fwd_1)
    model.transformer.out_proj.forward = _compile_cudagraph_wrap(model.transformer.out_proj.forward)

    return model

# ---------------------------------------------------------------------------------------------------------------------
# Model Prediction
# ---------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def simulate(model: MyriadStepByStep, batch: dict[str, torch.Tensor], num_traj: int, ensemble_size: int) -> torch.Tensor:
    ts = batch["t_poke"].to("cuda:0")
    ts = einops.rearrange(ts, "b (t n_traj) -> b t n_traj", n_traj=num_traj)
    ts = ts[:, :, 0]  # only give one time

    given_pos = batch["pos_poke"].to("cuda:0")  # [b, N+K, 2], already correct for predict_simulate

    camera_static = batch["camera_static"].to("cuda:0")
    img = batch["x"].to("cuda:0")

    if ensemble_size > 1:
        # expand inputs for ensemble
        img = img.repeat_interleave(ensemble_size, dim=0)
        ts = ts.repeat_interleave(ensemble_size, dim=0)
        given_pos = given_pos.repeat_interleave(ensemble_size, dim=0)
        camera_static = camera_static.repeat_interleave(ensemble_size, dim=0)
    
    d_img = model.embed_image(img)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        simulation = model.predict_simulate(
            n_traj=num_traj,
            ts=ts,
            given_pos=given_pos,
            camera_static=camera_static,
            d_img=d_img,
        )
    
    if ensemble_size > 1:
        simulation = einops.rearrange(
            simulation,
            "(b e) t n_traj c -> b e t n_traj c",
            e=ensemble_size
        )
    return simulation


# ---------------------------------------------------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------------------------------------------------

@click.command()
@click.option("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file containing the model weights and config.")
@click.option("--example_name", type=str, default="falling_ball", help="Which example to run. Options: " + ", ".join(EXAMPLES.keys()))
@click.option("--mode", type=str, default="fully_poked", help="Which poking mode to use. Options: fully_poked, first_query, unpoked, background_fixed, object_coherence")
@click.option("--num_steps", type=int, default=32, help="Number of future steps to simulate.")
@click.option("--ensemble_size", type=int, default=256, help="Number of trajectories to sample for each query.")
@click.option("--out_dir", type=str, default="eval_results/qual", help="Directory to save the output visualizations.")
@click.option("--tag", type=str, default=None, help="Optional tag to add to the output directory for better organization.")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
def main(
    checkpoint_path: str,
    example_name: str = "falling_ball",
    mode: str = "first_query",
    num_steps: int=32,
    ensemble_size: int = 256,
    out_dir: str = "eval_results/qual",
    tag: str | None = None,
    seed: int = 42,
):
    seed_everything(seed)

    # get model
    model = get_model("openset", checkpoint_path)
    # get data for current example
    img, pts, num_traj = get_example(example_name, mode)
    num_known = pts.shape[1] - num_traj  # first num_known trajectories have a given poke
    batch = get_batch(img, pts, num_traj, num_steps)

    # predict future
    pred_sim = simulate(model, batch, num_traj, ensemble_size)

    # save results
    output_path = Path(out_dir)
    if tag is not None:
        output_path = output_path / tag
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_path / example_name / mode / now
    output_path.mkdir(parents=True, exist_ok=True)
    for idx in range(pred_sim.shape[1]):
        sim = pred_sim[0, idx]
        c_sim = sim.detach().cpu().numpy()
        c_sim[..., 0] *= img.width
        c_sim[..., 1] *= img.height

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)

        pt_colors = ["black"] * num_known + ["dimgrey"] * (num_traj - num_known)
        ax.scatter(c_sim[0, :, 0], c_sim[0, :, 1], s=250, color=pt_colors, edgecolors='green', linewidths=1.5, alpha=0.9)
        for n in range(c_sim.shape[1]):
            ax.plot(c_sim[:, n, 0], c_sim[:, n, 1], color="green", linewidth=5., alpha=0.95)
        
        ax.set_xlim(0, img.width)
        ax.set_ylim(img.height, 0)
        ax.axis('off')

        fig.savefig(output_path / f"{idx}.pdf", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"Saved to {output_path / f'{idx}.pdf'}")

    print("Done!")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = max(256, torch._dynamo.config.cache_size_limit)

    main()
