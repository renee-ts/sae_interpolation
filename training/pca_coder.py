import json
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

import einops
import torch
from natsort import natsorted
from safetensors.torch import load_model, save_model
import torch

class PCA_Coder(torch.nn.Module):
    def __init__(
            self,
            d_in: int,
            config: dict,
            device: str,
            dtype: torch.dtype,
    ):
        self.d_in = d_in
        self.config = config
        self.device = device
        self.dtype = dtype

        self.rank = config.get("rank", None)
        self.update_rule = config.get("update_rule", "impute")

        assert self.rank is not None, "Rank must be specified in the config."
        assert self.rank > 0, "Rank must be greater than 0."

        self.U = None
        self.S = None
        self.Vh = None

        self.mean = None

    @property
    def device(self):
        return self.device
    
    @property
    def dtype(self):
        return self.dtype
    
    @torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_bf16_supported(),
    )
    def fit(self, x: torch.Tensor):
        """
        Fit the PCA model to the input tensor.
        """
        assert x.ndim == 2, "Input tensor must be 2D."
        assert x.shape[1] == self.d_in, f"Input tensor must have shape (N, {self.d_in})."

        # Center the data
        self.mean = torch.mean(x, dim=0)
        x = x - self.mean

        # Compute SVD
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        # Keep only the top 'rank' components
        if self.update_rule == "impute":
            self.U = U[:, :self.rank]
            self.S = S[:self.rank]
            self.Vh = Vh[:self.rank, :]
        elif self.update_rule == "ema":
            alpha = self.config.get("alpha", 0.9)
            self.U = (1 - alpha) * self.U + alpha * U[:, :self.rank]
            self.S = (1 - alpha) * self.S + alpha * S[:self.rank]
            self.Vh = (1 - alpha) * self.Vh + alpha * Vh[:self.rank, :]
        else:
            raise ValueError(f"Unknown update rule: {self.update_rule}")
    
    @torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_bf16_supported(),
    )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply PCA coding to the input tensor.
        """
        assert self.U is not None, "PCA model has not been trained yet."
        assert self.S is not None, "PCA model has not been trained yet."
        assert self.Vh is not None, "PCA model has not been trained yet."

        # Center the input data
        x = x - self.mean

        # Apply PCA transformation
        z = torch.matmul(x, self.U[:, :self.rank])

        reconstructed_x = torch.matmul(z, self.Vh[:self.rank, :]) + self.mean
        return reconstructed_x
    
    def save_to_disk(self, path: str):
        """
        Save the PCA model to disk.
        """
        assert self.U is not None, "PCA model has not been trained yet."
        assert self.S is not None, "PCA model has not been trained yet."
        assert self.Vh is not None, "PCA model has not been trained yet."

        data = {
            "U": self.U,
            "S": self.S,
            "Vh": self.Vh,
            "mean": self.mean,
            "rank": self.rank,
            "d_in": self.d_in,
            "dtype": str(self.dtype),
        }

        torch.save(data, path)
        
        with open(path / "cfg.json", "w") as f:
            json.dump(self.config, f, indent=4)
    
    def load_from_disk(self, path: str, device: str):
        """
        Load the PCA model from disk.
        """
        data = torch.load(path, map_location=device)
        self.U = data["U"].to(device)
        self.S = data["S"].to(device)
        self.Vh = data["Vh"].to(device)
        self.mean = data["mean"].to(device)

        assert self.rank == data["rank"], "Rank mismatch."
        assert self.d_in == data["d_in"], "Input dimension mismatch."
        assert self.dtype == torch.dtype(data["dtype"]), "Data type mismatch."
        
        with open(path / "cfg.json", "r") as f:
            self.config = json.load(f)
