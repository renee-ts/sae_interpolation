import torch
import logging
import os
import sys
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Any
from sparsify import Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize
from utils.model_utils import load_pythia_checkpoint
from utils.sae_utils import load_sae
from datasets import load_dataset


def train_sae(checkpoint_step: int, sae_config: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_pythia_checkpoint(checkpoint_step, device=device)
    if model is None or tokenizer is None:
        return
        
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    tokenized = chunk_and_tokenize(dataset, tokenizer)

    sae_cfg = load_sae(sae_config)
    
    train_config = TrainConfig(
        sae=sae_cfg,
        batch_size=8,
        optimizer="adam",
        run_name=f"Pythia-160M-step-{checkpoint_step}",
        save_best=True
    )
    train_config.save_dir = sae_config["output_dir"]
    os.makedirs(train_config.save_dir, exist_ok=True)
    trainer = None
    try:
        trainer = Trainer(train_config, tokenized, model)
        trainer.fit()
        trainer.save()
    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        if trainer is not None:
            del trainer
        torch.cuda.empty_cache()
        gc.collect()
    
    del model, tokenizer, tokenized
    torch.cuda.empty_cache()
    gc.collect()
