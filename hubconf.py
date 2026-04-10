# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025 Stefan Baumann et al., CompVis @ LMU Munich

import torch


dependencies = ["torch", "einops", "jaxtyping", "transformers", "torchvision"]


def fpt_base(*, pretrained: bool = True, **kwargs):
    """
    Standard Flow Poke Transformer.
    """
    from flow_poke.model import FlowPokeTransformer_Base

    model = FlowPokeTransformer_Base(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/CompVis/flow-poke-transformer/resolve/main/flow_poke_open_set_base.pt",
            map_location="cpu",
        )
        model.load_state_dict(state_dict["model"])
    model.requires_grad_(False)
    model.eval()
    return model

def myriad_openset(*, pretrained: bool = True, **kwargs):
    """
    Open-Set MYRIAD model
    """
    from myriad.model import MyriadStepByStep_Large

    model = MyriadStepByStep_Large(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/CompVis/myriad/blob/main/myriad_openset.pt",
            map_location="cpu",
        )
        model.load_state_dict(state_dict["model"])
    model.requires_grad_(False)
    model.eval()
    return model

def myriad_billiard(*, pretrained: bool = True, **kwargs):
    """
    Billiard MYRIAD model
    """
    from myriad.model import MyriadStepByStep_Large_Billiard

    model = MyriadStepByStep_Large_Billiard(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/CompVis/myriad/blob/main/myriad_billiard.pt",
            map_location="cpu",
        )
        model.load_state_dict(state_dict["model"])
    model.requires_grad_(False)
    model.eval()
    return model