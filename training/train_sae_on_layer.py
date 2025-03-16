import torch
import logging
import os
from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize
from utils.model_utils import load_pythia_checkpoint, extract_activations


def train_sae_on_layer(checkpoint_step, layer_idx, sae_config, dataset):
    """
    Trains an SAE on a single layer of a single model checkpoint.

    Note: in the Eleuther repo implementation, they call `Trainer(cfg, tokenized, gpt).
    It expects activations as input, but we want the activations from a specific layer.
    """
    logging.info(f"Training SAE on layer {layer_idx} at checkpoint {checkpoint_step}")

    model, tokenizer = None, None # TODO

    dataset = None # TODO

    tokenized = None # TODO

    activations = None # TODO

    config = TrainConfig(None, None) # TODO
    
    trainer = Trainer(config, activations, model)
    trainer.fit()

    layer_dir = f"./sae_checkpoints/layer_{layer_idx}"
    os.makedirs(layer_dir, exist_ok=True)
    sae_checkpoint_path = os.path.join(layer_dir, f"checkpoint_step_{checkpoint_step}.pt")

    torch.save(trainer.sae.state_dict(), sae_checkpoint_path)
    logging.info(
        f"SAE finished training for layer {layer_idx}, checkpoint {checkpoint_step}. "
        f"Saved state dict at: {sae_checkpoint_path}")
