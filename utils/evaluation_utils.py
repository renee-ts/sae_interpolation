from utils.sae_utils import load_sae_from_disk
from typing import Callable
from pathlib import Path
import torch

def reshaped_sae(sae) -> Callable:
    def wrapper(x: torch.Tensor) -> torch.Tensor:
        print(type(x))
        B, T, D = x.shape
        latents = sae(x.view(B * T, D)).latent_acts  # returns [B*T, n_latents]
        return latents.view(B, T, -1)    # reshape to [B, T, n_latents]
    return wrapper

def populate_submodule_dict(step_num: int, model_dim: int, sae_path: Path, device: int = "gpu") -> dict[str, Callable]:
    sae_layer_0 = load_sae_from_disk(sae_path / f"step-{step_num}" / f"step-{step_num}" / "best" / f"layers.0" / f"layers.0", d_in = model_dim, device = device)
    sae_layer_6 = load_sae_from_disk(sae_path / f"step-{step_num}" / f"step-{step_num}" / "best" / f"layers.6" / f"layers.6", d_in = model_dim, device = device)
    sae_layer_11 = load_sae_from_disk(sae_path / f"step-{step_num}" / f"step-{step_num}" / "best" / f"layers.11" / f"layers.11", d_in = model_dim, device = device)
    hookpoint = {"gpt_neox.layers.0.mlp": reshaped_sae(sae_layer_0), "gpt_neox.layers.6.mlp": reshaped_sae(sae_layer_6), "gpt_neox.layers.11.mlp": reshaped_sae(sae_layer_11)} 
    return hookpoint