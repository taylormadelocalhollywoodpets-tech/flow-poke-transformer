import gc
import os
import random
import math
from typing import Any, Iterator
import torch
import numpy as np
import einops
from jaxtyping import Float, Int
import cv2
import billiards

COLORS: dict[str, tuple[int, int, int]] = {
    "BACKGROUND": (
        255,
        255,
        255,
    ),  # white
    "BOUNDS": (
        128,
        128,
        128,
    ),  # gray
    "OUR_BALL": (
        255,
        0,
        0,
    ),  # red
    "DEFAULT_BALL": (
        0,
        0,
        0,
    ),  # black
}

# ---------------------------------------------------------------------------------------------------------------------
# General Data Utilities
# ---------------------------------------------------------------------------------------------------------------------

def worker_init_fn(worker_id):
    """
    Function to seed all underlying libraries in the workers
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True, **kwargs):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}  # remove keys with "__"

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = torch.from_numpy(np.array(list(batched[key])))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
            else:
                result[key] = list(batched[key])
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = torch.from_numpy(np.stack(list(batched[key])))
        else:
            result[key] = list(batched[key])
    return result

def dict_join_collation_fn(samples: list[dict[str, list | torch.Tensor]]):
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if torch.is_tensor(batched[key][0]):
            result[key] = torch.cat(batched[key])
        elif isinstance(batched[key][0], list):
            result[key] = sum(batched[key], [])
        else:
            raise ValueError(f"Unsupported type for key {key}: {type(batched[key][0])}")
    return result

class CollatingDataLoader:
    def __init__(self, inner_dataloader: torch.utils.data.DataLoader, collation_factor: int, collate_fn):
        self.inner_dataloader = inner_dataloader
        self.collation_factor = collation_factor
        self.collate_fn = collate_fn

    def __iter__(self):
        batch = []
        for data in self.inner_dataloader:
            batch.append(data)
            if len(batch) == self.collation_factor:
                yield self.collate_fn(batch)
                batch = []

# ---------------------------------------------------------------------------------------------------------------------
# Billiard Game Simulation
# ---------------------------------------------------------------------------------------------------------------------

def setup_billiard_game(
    frame_size: int,
    min_border_offset: int,
    max_border_offset: int,
    nr_balls: int,
    ball_radius: int,
    p_moving: float = 0.25,
) -> tuple[billiards.Billiard, list[int]]:
    border_offsets = [random.randrange(min_border_offset, max_border_offset) for _ in range(4)]
    bounds = [
        billiards.InfiniteWall(
            (0, border_offsets[0]),
            (frame_size - border_offsets[1], border_offsets[0]),
        ),
        billiards.InfiniteWall(
            (frame_size - border_offsets[1], border_offsets[0]),
            (frame_size - border_offsets[1], frame_size - border_offsets[2]),
        ),
        billiards.InfiniteWall(
            (frame_size - border_offsets[1], frame_size - border_offsets[2]),
            (border_offsets[3], frame_size - border_offsets[2]),
        ),
        billiards.InfiniteWall(
            (border_offsets[3], frame_size - border_offsets[2]),
            (border_offsets[3], border_offsets[0]),
        ),
    ]
    bld = billiards.Billiard(obstacles=bounds)
    num_balls = nr_balls  # random.randrange(2, nr_balls + 2)
    num_balls_added = 0
    border_margin = 0.1 * ball_radius
    while num_balls_added < num_balls:
        x = random.uniform(
            border_offsets[3] + ball_radius + border_margin,
            frame_size - border_offsets[1] - ball_radius - border_margin,
        )
        y = random.uniform(
            border_offsets[0] + ball_radius + border_margin,
            frame_size - border_offsets[2] - ball_radius - border_margin,
        )
        existing = getattr(bld, "balls_position", getattr(bld, "balls_initial_position", []))
        if any((x - bx) ** 2 + (y - by) ** 2 < (2 * ball_radius) ** 2 for bx, by in existing):
            continue

        if random.random() <= p_moving or num_balls_added == 0:
            bld.add_ball((x, y), (random.gauss(0, frame_size / 2), random.gauss(0, frame_size / 2)), ball_radius)
        else:
            bld.add_ball((x, y), (0, 0), ball_radius)
        num_balls_added += 1
    return bld, border_offsets


def simulate_billiard_game(
    bld: billiards.Billiard,
    duration: float,  # in seconds
    dt: float,  # in seconds,
) -> tuple[list, list, list]:
    start_time = bld.time
    frames = int(duration / dt) + 1
    ts = []
    pos = []
    vel = []
    collisions = []
    for i in range(frames):
        ball_ball_collisions, ball_obs_collisions = bld.evolve(start_time + i * dt)
        ts.append(bld.time)
        pos.append(bld.balls_position.copy())
        vel.append(bld.balls_velocity.copy())
        collisions.append(ball_ball_collisions > 0 or ball_obs_collisions > 0)
    return ts, pos, vel, collisions

# ---------------------------------------------------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------------------------------------------------

def render_billiard_frame(
    ball_pos: list[tuple[int, int]],
    ball_rad: list[int],
    frame_size: int,
    border_offsets: list[int],
    antialiasing: bool = False,
    goal_pos: tuple[int, int] | None = None,
) -> Int[np.ndarray, "h w c"]:
    assert len(border_offsets) == 4
    base_frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
    base_frame[:, :] = COLORS["BOUNDS"]
    cv2.rectangle(
        base_frame,
        (border_offsets[3], border_offsets[0]),
        (frame_size - border_offsets[1], frame_size - border_offsets[2]),
        COLORS["BACKGROUND"],
        -1,
    )
    for i_b, (b_p, b_r) in enumerate(zip(ball_pos, ball_rad)):
        cv2.circle(
            base_frame,
            (int(b_p[0]), int(b_p[1])),
            int(b_r),
            COLORS["OUR_BALL"] if i_b == 0 else COLORS["DEFAULT_BALL"],
            -1,
            lineType=cv2.LINE_AA if antialiasing else cv2.LINE_8,
        )
    
    if goal_pos is not None:
        x, y = int(goal_pos[0]), int(goal_pos[1])
        cross_size = 6  # half-length of each cross arm
        thickness = 2
        lt = cv2.LINE_AA if antialiasing else cv2.LINE_8

        # horizontal arm
        cv2.line(
            base_frame,
            (x - cross_size, y),
            (x + cross_size, y),
            (255, 0, 0),  # blue in BGR
            thickness,
            lineType=lt,
        )
        # vertical arm
        cv2.line(
            base_frame,
            (x, y - cross_size),
            (x, y + cross_size),
            (255, 0, 0),
            thickness,
            lineType=lt,
        )
    
    return base_frame

# ---------------------------------------------------------------------------------------------------------------------
# Billiard Dataset
# ---------------------------------------------------------------------------------------------------------------------

class BilliardSimDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        frame_size: int = 512,
        border_offset_range: list[float, float] = [0.1, 0.2],
        ball_radius: int = 0.033,
        nr_balls: int = 16,
        duration: float = 1.0,
        dt: float = 0.05,
        p_moving: float = 0.25,
        return_full_video: bool = False,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.ball_radius = ball_radius * frame_size
        self.nr_balls = nr_balls
        self.duration = duration
        self.dt = dt
        self.p_moving = p_moving
        self.return_full_video = return_full_video

        border_offset_range = np.array(border_offset_range)
        self.min_border_offset = max(math.floor(border_offset_range.min() * frame_size), 0)
        self.max_border_offset = min(math.ceil(border_offset_range.max() * frame_size), self.frame_size - 1)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            bld, border_offsets = setup_billiard_game(
                self.frame_size,
                self.min_border_offset,
                self.max_border_offset,
                self.nr_balls,
                self.ball_radius,
                self.p_moving,
            )
            ts, pos, vel, collisions = simulate_billiard_game(bld, self.duration, self.dt)

            if not self.return_full_video:
                frame: Int[np.ndarray, "h w c"] = render_billiard_frame(
                    pos[0], bld.balls_radius, self.frame_size, border_offsets
                )
                del bld
                gc.collect()
                x = einops.rearrange(torch.from_numpy(frame) / 127.5 - 1, "h w c -> c h w")
                x_cross = x.clone()
            else:
                frames = np.array([render_billiard_frame(p, bld.balls_radius, self.frame_size, border_offsets) for p in pos[:-1]])
                del bld
                gc.collect()
                x = einops.rearrange(torch.from_numpy(frames) / 127.5 - 1, "t h w c -> t c h w")
                x_cross = x[0].clone()

            pos_norm = np.array(pos, dtype=np.float32) / self.frame_size
            flow = pos_norm[1:] - pos_norm[:-1]
            pos_norm = pos_norm[:-1]

            T, N_t, _ = flow.shape
            pos_orig: Float[torch.Tensor, "t n_t 2"] = einops.repeat(pos_norm[0], "n_t c -> t n_t c", t=T)
            t: Float[torch.Tensor, "t n_t"] = einops.repeat(
                torch.arange(T).float(), "t -> t n_t", n_t=N_t
            )
            ids: Int[torch.Tensor, "t n_t"] = einops.repeat(torch.arange(N_t), "n_t -> t n_t", t=T)
            camera_static = torch.tensor(True)

            result = {
                "x": x,  # [C, H, W]
                "x_cross": x_cross,  # [C, H, W]

                "pos_poke": einops.rearrange(pos_norm, "t n_t c -> (t n_t) c"),
                "flow_poke": einops.rearrange(flow, "t n_t c -> (t n_t) c"),
                "pos_orig_poke": einops.rearrange(pos_orig, "t n_t c -> (t n_t) c"),
                "t_poke": einops.rearrange(t, "t n_t -> (t n_t)"),
                "id_poke": einops.rearrange(ids, "t n_t -> (t n_t)"),

                "pos_query": einops.rearrange(pos_norm, "t n_t c -> (t n_t) c"),
                "flow_query": einops.rearrange(flow, "t n_t c -> (t n_t) c"),
                "pos_orig_query": einops.rearrange(pos_orig, "t n_t c -> (t n_t) c"),
                "t_query": einops.rearrange(t, "t n_t -> (t n_t)"),
                "id_query": einops.rearrange(ids, "t n_t -> (t n_t)"),

                "camera_static": camera_static,

                "timeskip": self.dt,
                "border_offsets": torch.tensor(border_offsets),
                "collisions": torch.tensor(collisions, dtype=torch.bool),
            }
            yield result



# ---------------------------------------------------------------------------------------------------------------------
# Billiard DataModule
# ---------------------------------------------------------------------------------------------------------------------

class BilliardSimDataModule:

    def __init__(
        self,
        batch_size: int,
        train=None,
        validation=None,
        num_workers=4,
        val_batch_size: int | None = None,
        val_num_workers: int | None = None,
        pre_collation_factor: int = 1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.pre_collation_factor = pre_collation_factor
        if self.pre_collation_factor > 1:
            assert self.batch_size % self.pre_collation_factor == 0

    def make_loader(
        self,
        batch_size: int,
        num_workers: int,
        dataset_config: dict[str, Any],
        worker_init: bool = True,
        pre_collation_factor: int = 1,
    ):
        loader = torch.utils.data.DataLoader(
            BilliardSimDataset(**dataset_config),
            batch_size=batch_size // pre_collation_factor,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn if worker_init else None,
            collate_fn=dict_collation_fn,
            pin_memory=True,
        )
        if pre_collation_factor > 1:
            loader = CollatingDataLoader(loader, pre_collation_factor, dict_join_collation_fn)
        return loader

    def train_dataloader(self):
        return self.make_loader(
            **self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pre_collation_factor=self.pre_collation_factor,
        )

    def val_dataloader(self):
        return self.make_loader(
            **self.validation,
            batch_size=self.val_batch_size,
            num_workers=self.val_num_workers
        )
