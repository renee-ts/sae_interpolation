import torch
import os
import sys
import gc
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Any, Optional, List
from sparsify import Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize
from utils.model_utils import load_pythia_checkpoint
from utils.sae_utils import load_sae
from datasets import load_dataset


def train_sae(checkpoint_step: int, sae_config: Dict[str, Any], layers: Optional[List[int]]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_pythia_checkpoint(checkpoint_step, device=device)
    if model is None or tokenizer is None:
        return
        
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    tokenized = chunk_and_tokenize(dataset, tokenizer)

    sae_cfg = load_sae(sae_config)
    
    if layers:
        train_config = TrainConfig(
            sae=sae_cfg,
            batch_size=4,
            optimizer="adam",
            run_name=f"step-{checkpoint_step}",
            dead_feature_threshold=float('inf'),
            save_every=1000,
            save_best=True,
            log_to_wandb=False,
            layers=layers
        )   
    else:
        train_config = TrainConfig(
            sae=sae_cfg,
            batch_size=1,
            optimizer="adam",
            run_name=f""
        )

    save_dir_path = os.path.join(sae_config["output_dir"], f"step-{checkpoint_step}")
    train_config.save_dir = save_dir_path
    os.makedirs(train_config.save_dir, exist_ok=True)
    print(f"Saving checkpoint to: {train_config.save_dir}")
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

    return train_config.save_dir
