import gc
import click
import einops
import torch
import random
import numpy as np
import pandas as pd
import mediapy as mp

from datetime import datetime
from tabulate import tabulate
from pathlib import Path
from functools import partial
from tqdm import trange

from .qual import get_model
from myriad.model import MyriadStepByStep

DEVICE = "cuda"
MYRIAD_DTYPE = torch.float32
DTYPE = torch.bfloat16


# =============== Utility Functions ===============


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def free_mem():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


# =============== Metrics ===============


def topk_mse(preds: torch.Tensor, target: torch.Tensor, k: int = 5):
    """
    preds: (B, E, T, N, 2)
    target: (B, T, N, 2)
    """
    assert preds.ndim == 5 and target.ndim == 4
    B, E, T, N, C = preds.shape
    assert target.shape == (B, T, N, C) and C == 2

    diff = preds - target[:, None, ...]
    se = (diff ** 2).sum(dim=-1)
    denom = torch.tensor(T * N, dtype=se.dtype, device=se.device).view(1, 1).expand(B, 1)

    mse = se.sum(dim=(-1, -2)) / denom

    topk_vals_neg, topk_idx = torch.topk(-mse, k=min(k, E), dim=1)
    topk_vals = - topk_vals_neg

    return mse, topk_vals, topk_idx


# =============== MYRIAD ===============


@torch.no_grad()
def myriad_predict_function_with_past_cond(row, model: MyriadStepByStep, ensemble_size, gt_tracks, traj_kind, dt_scaling_factor, context_length=1, **kwargs):
    assert context_length == 1, f"Should eval with single-timestep poke conditioning, but found {context_length=}"
    gt_tracks = gt_tracks[traj_kind]["tracks"]
    num_trajectories = gt_tracks.shape[1]
    start_frame = row["start_frame"] - 1
    timesteps = row["context_end"] - start_frame
    total_video_length = row["frames"].shape[0]
    fps = row["fps"]

    gt_tracks_tensor = torch.from_numpy(gt_tracks[..., [1, 0]]).to(DEVICE).to(MYRIAD_DTYPE) / 2 + 0.5
    gt_tracks_before_start = gt_tracks_tensor[:start_frame, ...]
    gt_tracks_for_poke_conditioning = gt_tracks_tensor[start_frame:start_frame+2]
    assert gt_tracks_for_poke_conditioning.shape == (2, num_trajectories, 2), f"{gt_tracks_for_poke_conditioning.shape=}"

    flow_single = (gt_tracks_for_poke_conditioning[1] - gt_tracks_for_poke_conditioning[0])
    pts = torch.stack([gt_tracks_for_poke_conditioning[0], gt_tracks_for_poke_conditioning[0] + flow_single], dim=0).unsqueeze(0).to(DEVICE)
    assert pts.shape == (1, 2, num_trajectories, 2), f"expected {(1, 2, num_trajectories, 2)}, got {pts.shape=}"
    pts = einops.rearrange(pts, "b t n c -> b (t n) c")

    img = einops.rearrange(prepare_frames_for_tapnext(row["frames"][start_frame], dtype=MYRIAD_DTYPE), "h w c -> 1 c h w")

    if ensemble_size > 1:
        pts = einops.repeat(pts, "b ... -> (b e) ...", e=ensemble_size)
        img = einops.repeat(img, "b ... -> (b e) ...", e=ensemble_size)

    ts = einops.repeat(torch.arange(timesteps, device=DEVICE).float(), "t -> b t", b=pts.size(0)).to(DEVICE)
    ts = ts * (1 / fps)        # NOTE: technically this should be ts * (downsampling_factor / fps) but this is not how the model was trained I think
    ts = ts * dt_scaling_factor

    camera_static = torch.ones((pts.size(0),), device=DEVICE, dtype=torch.bool)
    bf16 = kwargs.get("bf16", False)
    autocast_dtype = torch.bfloat16 if bf16 else torch.float32
    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
        d_img = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in model.embed_image(img).items()
        }
        simulation = model.predict_simulate(
            n_traj=num_trajectories,
            ts=ts,
            given_pos=pts,
            camera_static=camera_static,
            d_img=d_img,
            verbose=False,
            d_steps=kwargs.get("d_steps", None)
        )

    if ensemble_size > 1:
        simulation = einops.rearrange(simulation, "(1 e) ... -> e ...", e=ensemble_size)
        gt_tracks_before_start = einops.repeat(gt_tracks_before_start, "... -> e ...", e=ensemble_size)
    else:
        simulation = simulation.unsqueeze(1)
        gt_tracks_before_start = gt_tracks_before_start.unsqueeze(1).unsqueeze(1)

    simulation = torch.cat([gt_tracks_before_start, simulation], dim=1)
    assert simulation[0].shape == gt_tracks_tensor.shape, f"{simulation[0].shape=} {gt_tracks_tensor.shape=}"
    assert simulation.shape == (ensemble_size, total_video_length, num_trajectories, 2), f"Expected {(ensemble_size, total_video_length, num_trajectories, 2)}, got {simulation.shape=}"

    simulation = simulation * 2 - 1

    return simulation.unsqueeze(0).cpu().numpy()[..., [1, 0]], einops.repeat(row["frames"], "t h w c -> b t h w c", b=ensemble_size), []


# =============== Dataset Loaders ===============


def load_idx_physion(data_root: Path, idx: int, labels: pd.DataFrame, context_frames: int = 16, video_length: int = 150):
    video_name = labels.iloc[idx]['name'].replace("_img", "")

    if 'contain' in video_name:
        task = 'Contain'
    elif 'collision' in video_name:
        task = 'Collide'
    elif 'dominoes' in video_name:
        task = 'Dominoes'
    elif 'drop' in video_name:
        task = "Drop"
    elif 'linking' in video_name:
        task = "Link"
    elif "rolling" in video_name:
        task = "Roll"
    elif "towers" in video_name:
        task = "Support"
    else:
        raise ValueError(f"Unknown task: {video_name}")

    ry_video_name = video_name.split("_")
    ry_video_name.insert(-1, "redyellow")
    ry_video_name = "_".join(ry_video_name) + "_img.mp4"
    ry_video_name = ry_video_name.replace("_redyellow", "-redyellow")

    video_path = data_root / task / "mp4s-redyellow" / ry_video_name
    frames = mp.read_video(str(video_path))

    video_base = Path(video_path).parent.parent

    map_path = video_base / "maps" / f"{video_name}_map.png"
    map_img = mp.read_image(map_path)

    label = labels.iloc[idx]["ground truth outcome"]
    start_frame = labels.iloc[idx]["start_frame"]
    map_vid = map_img[None].repeat(frames.shape[0], axis=0)

    (
        [frames_reduced_fps, map_vid_reduced_fps],
        start_frame,
        context_start,
        context_end
    ) = reduce_fps_cut_and_recalculate_start([frames, map_vid], start_frame, start_frame+video_length, context_frames, factor=2)

    return {
        # Physion-specific
        "map_vid": map_vid_reduced_fps,
        "label": label,

        # General
        "frames": frames_reduced_fps,
        "context_start": context_start,
        "start_frame": start_frame,
        "context_end": context_end,
        "video_path": video_path,
        "points": np.array(labels.iloc[idx]["points"]),
        "fps": 15,
    }


def load_idx_physics_iq(data_root: Path, idx: int, labels: pd.DataFrame, context_frames: int = 8, video_length: int = 80):
    conditioning_frames = mp.read_video(data_root / "physics-IQ-benchmark" / labels.iloc[idx]["conditioning_path"])
    testing_frames = mp.read_video(data_root / "physics-IQ-benchmark" / labels.iloc[idx]["testing_path"])
    start_frame = labels.iloc[idx]["start_frame"]

    H, W = conditioning_frames.shape[1:3]
    to_consider = min(H, W)
    scale = 512 / to_consider
    new_H, new_W = int(round(H * scale)), int(round(W * scale))
    conditioning_frames, testing_frames = mp.resize_video(conditioning_frames, (new_H, new_W)), mp.resize_video(testing_frames, (new_H, new_W))

    combined_frames = np.concatenate([conditioning_frames, testing_frames], axis=0)

    (
        [frames_reduced_fps],
        start_frame,
        context_start,
        context_end
    ) = reduce_fps_cut_and_recalculate_start([combined_frames], start_frame, start_frame+video_length, context_frames, factor=1)

    return {
        "frames": frames_reduced_fps,
        "context_start": context_start,
        "start_frame": start_frame,
        "context_end": context_end,
        "video_path": data_root / "physics-IQ-benchmark" / Path(labels.iloc[idx]["conditioning_path"]),
        "points": np.array(labels.iloc[idx]["points"]),
        "fps": 16,
    }


def load_idx_owm(data_root: Path, idx: int, labels: pd.DataFrame, context_frames: int = 8, video_length: int = 80):
    start_frame = labels.iloc[idx]["startFrame"]
    frames = mp.read_video(data_root / "video" / labels.iloc[idx]["name"])

    (
        [frames_reduced_fps],
        start_frame,
        context_start,
        context_end
    ) = reduce_fps_cut_and_recalculate_start([frames], start_frame, start_frame+video_length, context_frames, factor=1)

    return {
        "frames": frames_reduced_fps,
        "context_start": context_start,
        "start_frame": start_frame,
        "context_end": context_end,
        "video_path": Path(labels.iloc[idx]["name"]),
        "points": np.array(labels.iloc[idx]["points"]),
        "fps": frames.metadata.fps,
    }


# =============== Benchmark Utils ===============


def get_extra_query_pts_default(
    row,
    **kwargs,
):
    H, W = row["frames"].shape[1:3]
    grid = torch.rand((kwargs["num_random"], 2)) # [0, 1]
    grid = grid.mul(torch.tensor([W, H])[None, :]) # [0, size - 1]
    return {
        "random": grid.detach().cpu().numpy(),
    }


def reduce_fps_cut_and_recalculate_start(videos, start_frame, end_frame, context, factor=2):
    assert all(v.shape[0] == videos[0].shape[0] for v in videos), "All videos must have the same number of frames"

    context_start = max(0, start_frame - context)
    context_end = min(videos[0].shape[0], end_frame)

    videos = [v[context_start:context_end:factor] for v in videos]
    start_frame = (start_frame - context_start) // factor
    context_start = 0
    context_end = len(videos[0])

    return videos, start_frame, context_start, context_end


def prepare_frames_for_tapnext(frames, *, device=DEVICE, dtype=DTYPE):
    if isinstance(frames, torch.Tensor):
        return (frames / 255.0 * 2 - 1).to(device).to(dtype)
    elif isinstance(frames, np.ndarray):
        return torch.from_numpy(frames.astype(np.float32) / 255.0 * 2 - 1).to(device).to(dtype)


def prepare_points_for_tapnext(points):
    return points[..., [1, 0]]


def pin_xy_to_norm(points, *, W, H):
    return points / np.array([[W, H]])


def all_extra_query_points_except(query_points, *, keys: list[str], return_keys=False):
    if return_keys:
        return {k: query_points[k] for k in query_points if k not in keys}
    return [query_points[k] for k in query_points if k not in keys]


def resample_generated_to_conditioning(conditioning: np.ndarray, generated: np.ndarray):
    return mp.resize_video(generated, (conditioning.shape[1], conditioning.shape[2]))


def get_tapnext_cache(data_root: Path, video_path: Path):
    cache_path = data_root / "tapnext" / (video_path.stem + "_tapnext.npz")
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return {
            'tracks_yx.npy': data['tracks_yx.npy'],
            'logits_visible.npy': data['logits_visible.npy'],
            'certainty.npy': data['certainty.npy'],
            'query_points.npy': data['query_points.npy'],
            'query_points_splits.npy': data.get('query_points_splits.npy', None),
        }, True
    
    raise NotImplementedError("No Track annotations available")


def get_annotated_video(*, data_root: Path, row, get_extra_query_pts, override_cache = False, **kwargs):
    frames_tensor = einops.rearrange(prepare_frames_for_tapnext(row["frames"]).unsqueeze(0), "b t h w c -> b c t h w")
    H, W = frames_tensor.shape[3:]
    pin_to_frame_size = partial(pin_xy_to_norm, H=H-1, W=W-1)

    # get query points
    extra_query_points = get_extra_query_pts(row, **kwargs)
    extra_queries = [prepare_points_for_tapnext(pin_to_frame_size(x)) for x in all_extra_query_points_except(extra_query_points, keys=["random", "all"])]
    query_type_lens = [kwargs["num_bg"], *[x.shape[0] for x in extra_queries]]
    split_idxs = np.cumsum(query_type_lens)

    tapnext_dict, _ = get_tapnext_cache(
        data_root=data_root,
        video_path=row["video_path"],
    )

    tracks = tapnext_dict['tracks_yx.npy'][0]
    visibility = np.ones_like(tapnext_dict['logits_visible.npy'][0])
    certainty = np.ones_like(tapnext_dict['certainty.npy'][0])

    split_tracks = np.split(tracks, split_idxs, axis=1)
    split_visibilty = np.split(visibility, split_idxs, axis=1)
    split_certainty = np.split(certainty, split_idxs, axis=1)


    all_annotations = {
        "all": {
            "tracks": tracks,
            "visibility": visibility,
            "certainty": certainty,
        },
        "background": {
            "tracks": split_tracks[0],
            "visibility": split_visibilty[0],
            "certainty": split_certainty[0],
            "index": 0,
        },
        "annotated": {
            "tracks": split_tracks[-1],
            "visibility": split_visibilty[-1],
            "certainty": split_certainty[-1],
            "index": -1,
        },
        **{
            k: {
                "tracks": split_tracks[1 + i],
                "visibility": split_visibilty[1 + i],
                "certainty": split_certainty[1 + i],
                "index": 1 + i,
            }
            for i, k in enumerate(all_extra_query_points_except(extra_query_points, keys=["random", "all"], return_keys=True))
        }
    }
    return all_annotations, torch.from_numpy(tapnext_dict['query_points.npy']).to(DEVICE), tapnext_dict['query_points_splits.npy']

    
def evaluate_for_idx_generic(row, annotations, pred_sim, query_splits, cut_video_to_len):
    start_frame = row["start_frame"] - 1
    end_frame = start_frame + cut_video_to_len

    pred_sim = pred_sim[:, :, start_frame:end_frame, ...]
    pred_sim_split = np.split(pred_sim, query_splits, axis=-2)
    pred_annotated = torch.from_numpy(pred_sim_split[-1])

    gt_annotated = torch.from_numpy(annotations["annotated"]["tracks"]).unsqueeze(0)
    gt_annotated = gt_annotated[:, start_frame:end_frame, ...]

    mse, top_k_vals, top_k_idx = topk_mse(pred_annotated, gt_annotated)

    return {
        **{f"{i}_MSE": mse[0, i].item() for i in range(mse.shape[1])},
        "top_k_MSE": top_k_vals[0, 0].item(),
    }, top_k_idx


# =============== Main Benchmarking Script ===============


def benchmark(
    *,
    data_root: str,
    dataset_name: str,
    model,
    ensemble_size: int = 5,
    get_annotation_fn_kwargs: dict,
    pred_fn_kwargs: dict,
    out_path: Path,
    cut_video_to_len: int | None = None,
):
    if dataset_name == "owm":
        load_idx_fn = load_idx_owm
    elif dataset_name == "physion":
        load_idx_fn = load_idx_physion
    elif dataset_name == "physics-iq":
        load_idx_fn = load_idx_physics_iq
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    labels = pd.read_json(Path(data_root) / "annotations.json")

    metrics = []
    out_path = out_path / "partial_results"
    out_path.mkdir(parents=True, exist_ok=True)

    for idx in trange(len(labels)):
        try:
            row = load_idx_fn(data_root=data_root, idx=idx, labels=labels)

            seed_everything(42)
            annotations, _, query_splits = get_annotated_video(
                data_root=data_root,
                row=row,
                get_extra_query_pts=get_extra_query_pts_default,
                **get_annotation_fn_kwargs,
            )

            seed_everything(42)
            pred_sim, _, query_splits = myriad_predict_function_with_past_cond(
                row=row,
                model=model,
                ensemble_size=ensemble_size,
                gt_tracks=annotations,
                **pred_fn_kwargs,
            )

            # evals for all datasets
            generic_metrics, _ = evaluate_for_idx_generic(row, annotations, pred_sim, query_splits, cut_video_to_len)

            metrics.append({**generic_metrics, "idx": idx})
            pd.DataFrame([metrics]).to_json(out_path / f"{idx:02d}.json", orient="records", indent=4)
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
    
    return metrics


@click.command(context_settings={"show_default": True})
@click.option("--data-root", type=click.Path(dir_okay=True, file_okay=False, path_type=Path))
@click.option("--ckpt-path", type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
@click.option("--dataset-name", type=str, required=True, help="Dataset name we want to use.")
@click.option("--out-path", type=click.Path(dir_okay=True, file_okay=False, path_type=Path), default="./eval_results")
@click.option("--ensemble-size", default=5, show_default=True, type=int)
def main(
    data_root: Path,
    ckpt_path: Path,
    dataset_name: str,
    out_path: Path,
    ensemble_size: int,
):
    seed_everything(42)
    out_path = Path(out_path) / dataset_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path.mkdir(parents=True, exist_ok=True)

    model = get_model("openset", ckpt_path, compile_mode="max-autotune").to(MYRIAD_DTYPE)

    free_mem()
    metrics = benchmark(
        data_root = data_root,
        dataset_name = dataset_name,
        model = model,
        ensemble_size = ensemble_size,
        get_annotation_fn_kwargs=dict(
            num_bg = 10,
            num_random = 1024,
            jitter_scale = 3.5,
        ),
        pred_fn_kwargs=dict(
            traj_kind="annotated",
            dt_scaling_factor=30,
        ),
        out_path = out_path,
        cut_video_to_len = 32
    )
    
    metrics_df = pd.DataFrame(metrics)
    tk = metrics_df["top_k_MSE"]
    click.echo("\n\n Benchmark Results:")
    click.echo(f"top_k_MSE: {tk.mean():.6f}")


if __name__ == "__main__":

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 4096
    torch._dynamo.config.accumulated_recompile_limit = 4096
    torch.set_grad_enabled(False)
    torch._dynamo.config.cache_size_limit = max(2**15, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_recompile_limit = max(2**15, torch._dynamo.config.accumulated_recompile_limit)

    from einops._torch_specific import allow_ops_in_compiled_graph

    allow_ops_in_compiled_graph()

    main()
    click.echo("Done! 🥳")