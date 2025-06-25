

def load_pca(config):
    """
    Return an instance of SaeConfig.

    Args:
        config: Should just be the 'pca' key in config.yaml
    Returns:
        Dict: a dictionary with the specified hyperparameters.

    Note:
        create_sae expects the config dict passed in to only have the relevant
        configurations needed for sae. So one should first call `sae_config = config['sae]`.
    """
    rank = config["rank"]
    update_rule = config["update_rule"]
    alpha = config["alpha"] if update_rule == "ema" else None
    pca_config = {
        "rank": rank,
        "update_rule": update_rule,
        "alpha": alpha
    }

    return pca_config
