import torch
import logging
import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Any
from sparsify import TrainConfig
from .low_rank_trainer import LR_Trainer
from sparsify.data import chunk_and_tokenize
from utils.model_utils import load_pythia_checkpoint
from utils.pca_utils import load_pca
from datasets import load_dataset

@dataclass
class TrainPCAConfig(TrainConfig):
    pca: dict[str, Any]
    def __post_init__(self):
        super().__post_init__()


def train_pca(checkpoint_step: int, pca_config: Dict[str, Any], dataset_name: str):
    """
    Trains an SAE on a model.
    """
    dataset = load_dataset(dataset_name, split="train")

    if dataset is None:
        logging.error(f"Failed to load dataset: {dataset_name}.")
        return
    print("dataset loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer =  load_pythia_checkpoint(checkpoint_step, device=device)

    if model is None or tokenizer is None:
        logging.error(f"Failed to load model at checkpoint {checkpoint_step}.")
        return
    print("model loaded")

    tokenized = chunk_and_tokenize(dataset, tokenizer)
    print("tokenized")

    pca_cfg = load_pca(pca_config)
    num_layers =  model.config.num_hidden_layers
    hookpoints = hookpoints = [f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h" for layer_idx in range(num_layers)]
    
    train_config = TrainPCAConfig(pca=pca_cfg,
                               batch_size=16,
                               optimizer="adam",
                               run_name = "Pythia-410M",
                               hookpoints=hookpoints)

    print("train_config")

    trainer = LR_Trainer(train_config, tokenized, model)
    print("trainer done")

    trainer.fit()
    print("trainer fitted")
