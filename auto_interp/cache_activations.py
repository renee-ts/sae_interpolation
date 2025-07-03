from safetensors.torch import load_model
from datasets import load_dataset
from huggingface_hub import snapshot_download
from sparsify import SparseCoder, SparseCoderConfig
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch
import logging
from sparsify.data import chunk_and_tokenize
from typing import Callable
from delphi.config import CacheConfig
from delphi.latents import LatentCache
from utils.model_utils import load_pythia_checkpoint
from utils.evaluation_utils import populate_submodule_dict


sae_path = Path(snapshot_download(repo_id="sae-ml-drp/sae-ml-drp", repo_type= "model"))
print("SAE weights downloaded to:", sae_path)

# For now, we manually determine which checkpoint to extract activations from
checkpoint_step = 0 
model_dim = 768

cache_cfg = CacheConfig(
    dataset_repo = "EleutherAI/rpj-v2-sample", 
    dataset_split="train[:1%]",
    dataset_column="text",
    batch_size=8,
    cache_ctx_len=256,
    n_splits=5,
    n_tokens=10_000, # CHANGE, paper used n_tokens=200_000
)

model, tokenizer = load_pythia_checkpoint(checkpoint_step = checkpoint_step)
data = load_dataset(cache_cfg.dataset_repo, split=cache_cfg.dataset_split)
tokens = chunk_and_tokenize(data, tokenizer, max_seq_len=cache_cfg.cache_ctx_len, text_key="raw_content", num_proc = 1)["input_ids"]

# Create path for storing latent activations
latents_path = Path.cwd() / "my_latents"
latents_path.mkdir(parents=True, exist_ok=True)

# Create a log path within the run directory
log_path = latents_path.parent / "log"
log_path.mkdir(parents=True, exist_ok=True)

# Create dictionary with hookpoint-to-sae pairs
hookpoint_to_sparse_encoder = populate_submodule_dict(checkpoint_step, model_dim, sae_path, device = "cpu")

# Define and run cache to generate SAE activations
cache = LatentCache(
        model,
        hookpoint_to_sparse_encoder,
        batch_size=cache_cfg.batch_size,
        log_path=log_path,
        # Note: We set transcode = True because we use huggingface.AutoModelForCausalLM 
        # instead of a hooked transformer from transformerlens
        transcode = True 
    )
cache.run(cache_cfg.n_tokens, tokens)

# Save activations and config to `latents_path`
cache.save_splits(
        # Split the activation and location indices into different files to make
        # loading faster
        n_splits=cache_cfg.n_splits,
        save_dir=latents_path,
    )
cache.save_config(save_dir=latents_path, cfg=cache_cfg, model_name= f"pythia-160m-step-{checkpoint_step}")

