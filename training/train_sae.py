import torch
import logging
import os
from typing import Dict, Any
from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize
from utils.model_utils import load_pythia_checkpoint
from utils.sae_utils import load_sae
from utils.logging_utils import setup_logging
from datasets import load_dataset


def train_sae(checkpoint_step: int, sae_config: Dict[str, Any], dataset_name: str):
    """
    Trains an SAE on a single layer of a single model checkpoint.
    """
    dataset = None # TODO
    if dataset is None:
        logging.error(f"Failed to load dataset: {dataset_name}.")
        return

    model, tokenizer =  None, None # TODO
    if model is None or tokenizer is None:
        logging.error(f"Failed to load model at checkpoint {checkpoint_step}.")
        return

    tokenized = None # TODO

    sae_cfg  = None # TODO
    num_layers = None # TODO

    for layer_idx in range(num_layers):
        setup_logging(checkpoint_step=checkpoint_step, layer_idx=layer_idx)
        logging.info(f"Training SAE for layer {layer_idx} at checkpoint {checkpoint_step}.")
        
        train_config = None # TODO

        trainer = None # TODO
        trainer.fit()

        layer_dir = f"./sae_checkpoints/step_{checkpoint_step}/layer_{layer_idx}"
        os.makedirs(layer_dir, exist_ok=True)

        sae_checkpoint_path = os.path.join(layer_dir, "sae_checkpoint.pt")
        torch.save(trainer.sae.state_dict(), sae_checkpoint_path)

        logging.info(
            f"SAE finished training for layer {layer_idx}, checkpoint {checkpoint_step}. ")

    logging.info(f"Finished training all layers for checkpoint {checkpoint_step}.")
