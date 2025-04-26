import torch
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Any
from sparsify import Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize
from utils.model_utils import load_pythia_checkpoint
from utils.sae_utils import load_sae
from datasets import load_dataset


def train_sae(checkpoint_step: int, sae_config: Dict[str, Any]):
    """
    Trains an SAE on a model.
    """
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")

    if dataset is None:
        logging.error(f"Failed to load dataset: {dataset_name}.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer =  load_pythia_checkpoint(checkpoint_step, device=device)

    if model is None or tokenizer is None:
        logging.error(f"Failed to load model at checkpoint {checkpoint_step}.")
        return

    tokenized = chunk_and_tokenize(dataset, tokenizer)

    sae_cfg = load_sae(sae_config)
    num_layers =  model.config.num_hidden_layers
    hookpoints = hookpoints = [f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h" for layer_idx in range(num_layers)]
    
    train_config = TrainConfig(sae_cfg,
                               batch_size=4,
                               optimizer="adam",
                               run_name = "Pythia-410M",
                               hookpoints=hookpoints)


    trainer = Trainer(train_config, tokenized, model)

    trainer.fit()
