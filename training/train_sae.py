import torch
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    dataset = load_dataset(dataset_name, split="train")
    if dataset is None:
        logging.error(f"Failed to load dataset: {dataset_name}.")
        return
    print("dataset loaded")
    model, tokenizer =  load_pythia_checkpoint(checkpoint_step, device="cpu")
    if model is None or tokenizer is None:
        logging.error(f"Failed to load model at checkpoint {checkpoint_step}.")
        return
    print("model loaded")
    tokenized = chunk_and_tokenize(dataset, tokenizer)
    print("tokenized")
    sae_cfg = load_sae(sae_config)
    num_layers =  len(model.transformer.h)
    
    train_config = TrainConfig(sae_cfg, batch_size=32, optimizer="adam", run_name = "Pythia", hookpoint=[f"blocks.{layer_idx}.mlp_output" for layer_idx in (num_layers)]) # should this be hookpoints or layers
    print("train_config")
    trainer = Trainer(train_config, tokenized, model)
    print("trainer done")
    trainer.fit()
    print("tariner fitted")
