from sparsify import SaeConfig


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
    hidden_dim = config["hidden_dim"]
    k = config["k"]
    expansion_factor = config["expansion_factor"]
    activation = config["activation"]
    sae_config = SaeConfig(
        hidden_dim=hidden_dim,
        k=k,
        expansion_factor=expansion_factor,
        activation=activation
        )
    return sae_config
