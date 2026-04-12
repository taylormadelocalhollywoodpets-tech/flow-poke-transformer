import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import torchvision.transforms.functional as TF
import einops
from jaxtyping import Float

# ---------------------------------------------------------------------------------------------------------------------
# Wrapper for DINOv3 from HuggingFace.
# - We use DINOv3 as an image embedding model with optional finetuning.
# - Wrapper designed for compatibility with DINOv2 in Flow Poke Transformer repository
# ---------------------------------------------------------------------------------------------------------------------

class DinoV3HF(nn.Module):

    def __init__(
        self,
        model_size: int=256,
        model_version: str="dinov3_vitl16",
        gradient_last_blocks: None | int=None,
        reshape: bool=True,
        out: str="both",
        requires_grad: bool=False,
    ) -> None:
        super().__init__()

        self.model_size = model_size
        self.out = out
        self.reshape = reshape

        self.model = AutoModel.from_pretrained(model_version)

        if requires_grad:
            self.model.requires_grad_(True)
            self.model.train()
        else:
            self.model.requires_grad_(False)
            self.model.eval()

        self.num_register_tokens = self.model.config.num_register_tokens
        self.patch_size = self.model.config.patch_size
        
        self.gradient_last_blocks = gradient_last_blocks
        if gradient_last_blocks is not None and gradient_last_blocks > 0:
            blocks = self.model.encoder.layer
            for b in blocks[-gradient_last_blocks:]:
                b.requires_grad_(True)
                b.train()
    
    def _tokens_to_maps_and_cls(self, last_hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = last_hidden_state.shape
        
        cls = last_hidden_state[:, 0]  # B x D
        patches = last_hidden_state[:, 1+self.num_register_tokens:]  # B x (L-1) x D

        if not self.reshape:
            return patches, cls

        N = patches.shape[1]
        s = int(math.sqrt(N))
        if s * s != N:
            raise ValueError(f"Patch tokens {N} don't form a square; got sqrt={s}. Ensure square resize to {self.model_size}.")
        
        feats = einops.rearrange(patches, 'b (h w) d -> b d h w', h=s, w=s)
        return feats, cls
    
    def forward_features(self, imgs: Float[torch.Tensor, "B C H W"], masks) -> tuple[Float[torch.Tensor, "B D h' w'"], Float[torch.Tensor, "B D"]]:
        out = self.model(pixel_values=imgs)
        tokens = out.last_hidden_state  # B x L x D
        features, cls = self._tokens_to_maps_and_cls(tokens)
        return {
            "x_norm_patchtokens": einops.rearrange(features, "b d h w -> b (h w) d"),
            "x_norm_clstoken": cls,
        }
    
    def forward(self, imgs: Float[torch.Tensor, "B C H W"]) -> tuple[Float[torch.Tensor, "B D h' w'"], Float[torch.Tensor, "B D"]]:
        # Expect inputs scaled to [-1, 1]
        assert imgs.min() >= -1.0
        assert imgs.max() <= 1.0
        assert len(imgs.shape) == 4

        if not torch.jit.is_tracing():
            imgs = better_resize(imgs, self.model_size)

        imgs = (imgs + 1.0) / 2.0
        imgs = TF.normalize(imgs, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        out = self.model(pixel_values=imgs)
        tokens = out.last_hidden_state  # B x L x D

        features, cls = self._tokens_to_maps_and_cls(tokens)
        if self.out == "features":
            return features
        elif self.out == "class":
            return cls
        else:
            return features, cls

def dinov3_large():
    return DinoV3HF(
        model_size=512,
        model_version="facebook/dinov3-vitl16-pretrain-lvd1689m",
        out="features",
        gradient_last_blocks=None,
        reshape=True,
        requires_grad=True,
    )

def dinov3_base():
    return DinoV3HF(
        model_size=512,
        model_version="facebook/dinov3-vitb16-pretrain-lvd1689m",
        out="features",
        gradient_last_blocks=None,
        reshape=True,
        requires_grad=True,
    )

def dinov3_small():
    return DinoV3HF(
        model_size=512,
        model_version="facebook/dinov3-vits16-pretrain-lvd1689m",
        out="features",
        gradient_last_blocks=None,
        reshape=True,
        requires_grad=True,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Wrapper for extracting features
# ---------------------------------------------------------------------------------------------------------------------


def better_resize(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    ss = imgs.shape
    assert ss[-3] == 3

    H, W = ss[-2:]

    if len(ss) == 3:
        imgs = imgs.unsqueeze(0)

    side = min(H, W)
    imgs = TF.center_crop(imgs, [side, side])
    imgs = F.interpolate(imgs, [image_size, image_size], mode="bilinear", antialias=True)

    if len(ss) == 3:
        imgs = imgs[0]
    return imgs


class DinoFeatureExtractor(nn.Module):
    def __init__(
        self,
        image_size: int = 448,
        model_version: str = "dinov2_vitb14_reg",
        requires_grad: bool = False,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.image_size = image_size

        self.patch_size = 14 if "dinov2" in model_version else 16

        self.model = {
            "dinov3_vitl16": dinov3_large,
            "dinov3_vitl16_no_weights": dinov3_large,
            "dinov3_vitb16": dinov3_base,
            "dinov3_vits16": dinov3_small,
        }[model_version]()
        if "dinov2" in model_version:
            self.model.load_state_dict(original_model.state_dict(), strict=True)
        self.embed_dim = {
            "dinov3_vitl16": 1024,
            "dinov3_vitl16_no_weights": 1024,
            "dinov3_vitb16": 768,
            "dinov3_vits16": 384,
        }[model_version]

        if requires_grad:
            self.model.requires_grad_(True)
            self.model.train()
        else:
            self.model.requires_grad_(False)
            self.model.eval()

    def forward(self, imgs: Float[torch.Tensor, "b c h w"]) -> Float[torch.Tensor, "b c h' w'"]:
        assert imgs.min() >= -1.0
        assert imgs.max() <= 1.0
        assert len(imgs.shape) == 4

        if not torch.jit.is_tracing():
            imgs = better_resize(imgs, self.image_size)

        imgs = (imgs + 1.0) / 2.0  # [-1,1] -> [0,1]
        imgs = (imgs - torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, -1, 1, 1)) / torch.tensor(
            [0.229, 0.224, 0.225], device=imgs.device
        ).view(1, -1, 1, 1)
        d = self.model.forward_features(imgs, masks=None)

        features = einops.rearrange(
            d["x_norm_patchtokens"], "b (h w) d -> b d h w", h=self.image_size // self.patch_size, w=self.image_size // self.patch_size
        )
        return features
