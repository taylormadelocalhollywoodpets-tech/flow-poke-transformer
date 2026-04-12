import os
from pathlib import Path
import gc
import io
import time
import random
import json

import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.v2 as TVT
import av
import einops
import webdataset as wds
from jaxtyping import Float, Bool
import cv2
from functools import partial
import glob
from omegaconf import OmegaConf, ListConfig
from itertools import chain
from webdataset.filters import reraise_exception, pipelinefilter

from .data_billiards import dict_collation_fn

# ---------------------------------------------------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------------------------------------------------

def augment(
    clip,
    interpolation=TVT.InterpolationMode.BICUBIC,
    size=512,
):
    clip = TVT.functional.resize(clip, size, interpolation=interpolation, antialias=True)
    # normalize
    clip = (clip - 0.5) / 0.5
    clip = clip.clamp(-1.0, 1.0)  # to prevent values outside from [-1,1] in bicubic mode
    return clip

def decode_npy(b: bytes):
    with io.BytesIO(b) as f:
        return np.load(f, allow_pickle=True)

def _map_many(data, f, handler=reraise_exception):
    """Version of wds.map() that ."""
    for sample in data:
        try:
            results = f(sample)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        for i, r in enumerate(results):
            if r is None:
                continue
            if isinstance(sample, dict) and isinstance(r, dict):
                r["__key__"] = f"{sample.get('__key__', )}-{i}"
            yield r

map_many = pipelinefilter(_map_many)

# ---------------------------------------------------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------------------------------------------------

class TrackerShardsDataModule:

    def __init__(
        self,
        tar_base: str | list | ListConfig,
        batch_size: int,
        train=None,
        validation=None,
        num_workers=4,
        val_batch_size: int | None = None,
        val_num_workers: int | None = None,
        prefetch_factor: int = 8,
        num_tracks: int = 16,
        num_steps: int = 16,
        allow_invisible_track_ends: bool = True,
        allow_out_of_frame_track_ends: bool = False,
        filter_static_camera: bool = False,
        static_camera_flow_mag_threshold: float=0.00035,
        static_camera_fraction_threshold: float=0.2,
        return_full_sequence: bool=False,
        center_crop: bool=False,
        crop_size: int | None = None,  # provide an integer if we want to further center crop the image after resizing
        track_key: str = "tracks_yx",
        shuffle: bool = True,
        certainty_threshold: float = 0.6,
        visibility_threshold: float = 0.5,
        t_scaling_factor: float=30.0,
    ):
        super().__init__()
        self.tar_base = tar_base
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.prefetch_factor = prefetch_factor

        self.num_tracks = num_tracks
        self.num_steps = num_steps
        self.allow_invisible_track_ends = allow_invisible_track_ends
        self.allow_out_of_frame_track_ends = allow_out_of_frame_track_ends
        # self.flip_axes = flip_axes
        self.filter_static_camera = filter_static_camera

        self.static_camera_flow_mag_threshold = static_camera_flow_mag_threshold
        self.static_camera_fraction_threshold = static_camera_fraction_threshold

        self.return_full_sequence = return_full_sequence
        self.center_crop = center_crop
        self.crop_size = crop_size

        self.shuffle = shuffle
        self.track_key = track_key

        self.certainty_threshold = certainty_threshold
        self.visibility_threshold = visibility_threshold

        self.t_scaling_factor = t_scaling_factor

    def get_camera_static_tensor(self, flow: Float[torch.Tensor, "t n_t 2"]) -> Bool[torch.Tensor, ""]:
        # camera statix heuristic from Flow Poke Transformer
        return (
            flow.norm(dim=-1) < self.static_camera_flow_mag_threshold
        ).float().mean() > self.static_camera_fraction_threshold

    def _load_frame(self, video, i_frame: int):
        with io.BytesIO(video) as buf, av.open(buf) as container:
            c_f = 0
            target_frame = None
            for packet in container.demux():
                if not target_frame is None:
                    break
                for frame in packet.decode():
                    if c_f == i_frame:
                        target_frame = frame.to_ndarray(format="rgb24")
                        break
                    c_f += 1
        if target_frame is None:
            return None
        
        assert c_f == i_frame, f"{i_frame=}, {c_f=}"

        # numpy frame to normalized and resized tensor
        x: Float[torch.Tensor, "c h w"] = augment(
            einops.rearrange(torch.from_numpy(target_frame).float() / 255, "h w c -> 1 c h w"), size=512, center_crop=False
        )[
            0
        ].bfloat16()  # [-1, 1]
        return x
    
    def _center_crop(
        self,
        x: Float[torch.Tensor, "c h w"],
        pos: Float[torch.Tensor, "t l 2"],
        center_crop: bool,
        L: int | None = None,
    ):
        # center crop frames further and accordingly adjust tracks
        if not center_crop:
            return x, pos

        H, W = x.shape[-2], x.shape[-1]
        max_L = min(H, W)
        if L is None:
            L = max_L
        else:
            L = min(L, max_L)

        starth = (H - L) // 2
        startw = (W - L) // 2

        x = x[..., starth:starth + L, startw:startw + L]

        pos = pos.clone()
        offs = pos.new_tensor([startw, starth])
        scale = pos.new_tensor([W, H])

        pos_px = pos * scale
        pos_px = pos_px - offs
        pos = pos_px / L

        return x, pos

    def _load_adjusted_frame(
        self,
        video,
        i_frame: int,
        pos: Float[torch.Tensor, "t l 2"],
        visibility: Float[torch.Tensor, "t l"],
    ):
        x = self._load_frame(video, i_frame)
        x, pos = self._center_crop(x, pos, center_crop=self.center_crop, L=self.crop_size)
        # update visibility: pos < 0 or > 1 are out of frame
        track_in_frame: Bool[torch.Tensor, "t l"] = (
            (pos[:, :, 0] >= 0) & (pos[:, :, 0] <= 1) & (pos[:, :, 1] >= 0) & (pos[:, :, 1] <= 1)
        )
        return x, pos, visibility, track_in_frame

    def extract_training_sample(
        self,
        sample: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        assert all(
            k in sample
            for k in [
                "visibility",
                "tracks",
                "times",
                'certainty'
            ]
        ), f"{sample.keys()=}"

        try:
            tracks: Float[torch.Tensor, "t n_t 2"] = sample["tracks"]

            # visibility and certainty filtering
            visibility: Bool[torch.Tensor, "t n_t"] = sample["visibility"]
            certainty: Float[torch.Tensor, "t n_t"] = sample["certainty"]
            active = (visibility > self.visibility_threshold) & (certainty > self.certainty_threshold)

            track_in_frame: Bool[torch.Tensor, "t n_t"] = (
                (tracks[..., 0] >= 0) & (tracks[..., 0] <= 1) & (tracks[..., 1] >= 0) & (tracks[..., 1] <= 1)
            )
            visible_and_in_frame: Bool[torch.Tensor, "t n_t"] = active & track_in_frame
            valid_start_frames = (
                visible_and_in_frame.int().sum(dim=1) >= self.num_tracks
            )  # Ones that have at least `num_tracks` visible tracks

            if not valid_start_frames.any():
                return {"valid": False}

            valid_start_frames = valid_start_frames & (
                torch.arange(valid_start_frames.size(0)) <= valid_start_frames.size(0) - 1 - ((self.num_steps - 1))
            )  # Ones that have enough frames after them to sample a full sequence
            if not valid_start_frames.any():
                return {"valid": False}

            i_start = valid_start_frames.nonzero()[torch.randint(valid_start_frames.sum(), (1,))].squeeze()
            i_last = i_start + (self.num_steps - 1)

            valid_tracks_mask = visible_and_in_frame[i_start]
            if not self.allow_invisible_track_ends:
                valid_tracks_mask &= active[i_last]
            
            # Select random valid tracks
            valid_tracks = valid_tracks_mask.nonzero().flatten()
            if len(valid_tracks) < self.num_tracks:
                return {"valid": False}
            selected_track_idxs = valid_tracks[torch.randperm(len(valid_tracks))]
            pos: Float[torch.Tensor, "t l 2"] = tracks[i_start : i_last+1]

            # construct flow from pos
            flow: Float[torch.Tensor, "t-1 l 2"] = pos[1:] - pos[:-1]

            # detect before subsampling tracks
            camera_static = self.get_camera_static_tensor(flow)

            # NOTE: do not reduce number of tracks (yet!)
            pos = pos[:, selected_track_idxs]
            flow = flow[:, selected_track_idxs]

            visibility = visibility[i_start : i_last+1]
            visibility = visibility[:, selected_track_idxs]

            return {
                "i_frame": int(i_start.item()),
                "pos": pos,  # [t-1, l, 2] in [0, 1]
                "flow": flow,  # [t-1, l, 2] in ~[-1, 1]
                "timeskip": (sample["times"][i_last] - sample["times"][i_start]) / (self.num_steps - 1),  # in seconds
                "camera_static": camera_static,
                "visibility": visibility[:-1],  # [t-1, l]
            }
        except Exception as e:
            print(f"Error: {e=}")
            return {"valid": False}

    def _build_targets(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        T, N, C = sample["pos"].shape
        pos_orig = einops.repeat(sample["pos"][0], "n c -> t n c", t=T)

        frame_indices = torch.arange(T).float() * sample["timeskip"] * self.t_scaling_factor  # scale to match RoPE expectation
        t: Float[torch.Tensor, "t n_t"] = einops.repeat(frame_indices, "t -> t n", n=N)

        ids = einops.repeat(torch.arange(N, device=sample["pos"].device), "n -> t n", t=T)
        
        target_sample = {
            "pos_poke": einops.rearrange(sample["pos"], "t n c -> (t n) c"),
            "flow_poke": einops.rearrange(sample["flow"], "t n c -> (t n) c"),
            "pos_orig_poke": einops.rearrange(pos_orig, "t n c -> (t n) c"),
            "t_poke": einops.rearrange(t, "t n -> (t n)"),
            "id_poke": einops.rearrange(ids, "t n -> (t n)"),

            "pos_query": einops.rearrange(sample["pos"], "t n c -> (t n) c"),
            "flow_query": einops.rearrange(sample["flow"], "t n c -> (t n) c"),
            "pos_orig_query": einops.rearrange(pos_orig, "t n c -> (t n) c"),
            "t_query": einops.rearrange(t, "t n -> (t n)"),
            "id_query": einops.rearrange(ids, "t n -> (t n)"),
            "camera_static": sample["camera_static"],

            "i_frame": sample["i_frame"],
        }

        return target_sample

    def _decode(self, sample: dict[str, bytes]) -> dict[str, torch.Tensor]:
        try:
            assert all(
                k in sample
                for k in [
                    "video.mp4",
                    # "times.npy",
                    f"{self.track_key}.npy",
                    "logits_visible.npy",
                    "certainty.npy"
                ]
            ), f"{sample.keys()=}"

            tracks = decode_npy(sample[f"{self.track_key}.npy"])
            tracks = torch.from_numpy(tracks).float()
            tracks = (tracks + 1) / 2 # to [0, 1]
            tracks = tracks[..., [1, 0]] # (yx) to (xy)

            num_tracks_unfiltered = tracks.shape[1]
            certainty = torch.from_numpy(decode_npy(sample["certainty.npy"]))

            d = {
                "tracks": tracks,  # [t, n_t, 2]
                "visibility": torch.sigmoid(torch.from_numpy(decode_npy(sample["logits_visible.npy"]))),  # [t, n_t]
                "certainty": certainty,
            }

            # get fps and real time
            got_fps = False
            if "meta.json" in sample.keys():
                meta_bytes = sample["meta.json"]
                meta_str = meta_bytes.decode("utf-8")
                meta = json.loads(meta_str)
                if "fps" in meta.keys():
                    fps = float(meta["fps"])
                    got_fps = True
            if "fps" in sample.keys() and not got_fps:
                fps = float(sample["fps"])
                got_fps = True
            if not got_fps:
                fps = 30
            d["times"] = torch.arange(d["tracks"].shape[0]) / fps

            # get sample and valid tracks
            sample_out = self.extract_training_sample(d)
            if not sample_out.get("valid", True):
                return {"valid": False}

            # camera static filtering
            camera_static = sample_out["camera_static"]
            if self.filter_static_camera and not camera_static.item():
                return {"valid": False}

            # get frame
            i_f = sample_out.get("i_frame", 0) if not self.return_video_without_cutting else 0
            x, pos, visibility, track_in_frame = self._load_adjusted_frame(sample["video.mp4"], i_f, sample_out["pos"], sample_out["visibility"])

            if x is None:
                return {"valid": False}

            # load full video
            if self.return_full_sequence:
                loaded_frames = [x]
                end = i_f + self.num_steps
                for c_frame in range(i_f + 1, end):
                    c, _, _, _ = self._load_adjusted_frame(sample["video.mp4"], c_frame, sample_out["pos"], sample_out["visibility"])
                    if c == None:
                        break
                    loaded_frames.append(c)
                if len(loaded_frames) == 0:
                    return {"valid": False}
                x = torch.stack(loaded_frames, dim=0)[:-1]  # (t, c, h, w)

            # refilter by updated visibility
            valid_tracks_mask = (visibility[0] > self.visibility_threshold).nonzero().flatten()
            valid_tracks_mask = valid_tracks_mask[(track_in_frame[0, valid_tracks_mask])]
            if not self.allow_invisible_track_ends:
                valid_tracks_mask = valid_tracks_mask[(visibility[-1, valid_tracks_mask] > self.visibility_threshold)]
            if not self.allow_out_of_frame_track_ends:
                valid_tracks_mask = valid_tracks_mask[(track_in_frame[-1, valid_tracks_mask])]

            final_num_valid_tracks = len(valid_tracks_mask)
            if final_num_valid_tracks < self.num_tracks:
                return {"valid": False}

            selected_track_idxs = valid_tracks_mask[torch.randperm(len(valid_tracks_mask))[: self.num_tracks]]
            pos = pos[:, selected_track_idxs]
            visibility = visibility[:, selected_track_idxs]

            # recompute flow after cropping
            pos_f = pos.float()
            if self.delta_0_flow:
                flow = pos_f[1:] - pos_f[0:1]
            else:
                flow = pos_f[1:] - pos_f[:-1]
            flow = flow.to(pos.dtype)

            # reduce to num_tracks
            pos = pos[:-1, : self.num_tracks]
            visibility = visibility[:, : self.num_tracks]
            flow = flow[:, : self.num_tracks]

            assert torch.all((pos[0] >= 0) & (pos[0] <= 1)).item(), "pos out of frame at t=0"

            sample_out["pos"] = pos
            sample_out["visibility"] = visibility
            sample_out["flow"] = flow

            sample_out = self._build_targets(sample_out)

            return sample_out | {
                "x": x,
                "filtering_yield": torch.tensor([final_num_valid_tracks / num_tracks_unfiltered], dtype=torch.float32),
            }
        except Exception as e:
            print(f"Error: {e=}")
            return {"valid": False}

    def _filter_valid(self, sample: dict[str, torch.Tensor]) -> bool:
        valid = sample.get("valid", True)
        return valid

    def make_loader(
        self,
        shards: str,
        batch_size: int,
        num_workers: int,
        shuffle: int = 0,
    ):
        # resolve base directories
        base_dirs = self.tar_base
        if isinstance(base_dirs, (str, Path)):
            base_dirs = [base_dirs]
        elif isinstance(base_dirs, ListConfig):
            base_dirs = OmegaConf.to_object(base_dirs)
        elif isinstance(base_dirs, list):
            base_dirs = base_dirs
        else:
            raise NotImplementedError(f'Tar base is of type {type(base_dirs)} which is not supported as of now')

        base_dirs = [Path(b).expanduser().resolve() for b in base_dirs]

        # search for shards
        if shards is None:
            shard_urls = [
                str(p) for base in base_dirs for p in base.rglob("*.tar")
            ]
        else:
            if isinstance(shards, ListConfig):
                shards = OmegaConf.to_object(shards)
            if isinstance(shards, (list, tuple)):
                patterns = shards
            else:                             # a single string
                patterns = [shards]

            shard_urls = []
            for base in base_dirs:
                for pat in patterns:
                    full_pat = str(base / pat)
                    matches = glob.glob(full_pat)
                    shard_urls.extend(matches)
        
        if len(shard_urls) == 0:
            raise FileNotFoundError("No shards matched patterns")

        shard_urls = list(set(shard_urls))  # deduplicate
        shard_urls.sort()   # sort

        # data pipeline
        dataset = wds.DataPipeline(
            wds.SimpleShardList(shard_urls),
            wds.detshuffle() if self.shuffle else lambda x: x,
            wds.split_by_node,
            wds.split_by_worker,
            partial(wds.tarfile_samples, handler=wds.warn_and_continue),
            *([wds.shuffle(shuffle)] if shuffle != 0 and self.shuffle else []),
            wds.map(self._decode),
            wds.select(self._filter_valid),
            wds.batched(batch_size, partial=False, collation_fn=dict_collation_fn),
        )

        return wds.WebLoader(
            dataset, batch_size=None, num_workers=num_workers, prefetch_factor=self.prefetch_factor, pin_memory=True, persistent_workers=True
        )
    
    def train_dataloader(self):
        return self.make_loader(
            **self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return self.make_loader(
            **self.validation,
            batch_size=self.val_batch_size,
            num_workers=self.val_num_workers
        )
