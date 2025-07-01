from sparsify import SaeConfig, SparseCoderConfig, SparseCoder
from safetensors.torch import load_model
import Path



def load_sae(config):
    """
    Return an instance of SaeConfig.

    Args:
        config: Should just be the 'sae' key in config.yaml
    Returns:
        SAEConfig: a SAEConfig with the specified hyperparameters.

    Note:
        create_sae expects the config dict passed in to only have the relevant
        configurations needed for sae. So one should first call `sae_config = config['sae]`.
    """
    k = config["k"]
    expansion_factor = config["expansion_factor"]
    activation = config["activation"]

    sae_config = SaeConfig(
        k=k,
        expansion_factor=expansion_factor,
        activation=activation
        )

    return sae_config


def load_sae_from_disk(path: str, d_in: int, device: str = "gpu"):
    """
    Loads SAE from sae.safetensors file.

    Args:
        path: Path to directory containing sae.safetensors
        d_in: Dimension of input to SAE
        device: Defaults to gpu

    Returns:
        sae: An instance of Sparse Coder loaded from disk

    Note:
        
    """
    path = Path(path)
    cfg = SparseCoderConfig(
        expansion_factor=64,
        k=32,
        )
    sae = SparseCoder(d_in, cfg, device=device)
    load_model(
        model=sae,
        filename=str(path / "sae.safetensors"),
        device=str(device),
    )
    return sae
