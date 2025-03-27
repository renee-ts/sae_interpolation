import torch
import logging
import os
from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize
from utils.model_utils import load_pythia_checkpoint, extract_activations
from utils.sae_utils import load_sae


def train_sae_on_layer(checkpoint_step, layer_idx, sae_config, dataset_path):
    """
    Trains an SAE on a single layer of a single model checkpoint.

    Note: in the Eleuther repo implementation, they call `Trainer(cfg, tokenized, gpt).
    It expects activations as input, but we want the activations from a specific layer.
    """
    logging.info(f"Training SAE on layer {layer_idx} at checkpoint {checkpoint_step}")

    model, tokenizer = load_pythia_checkpoint(checkpoint_step)
    if model is None or tokenizer is None:
        logging.error(f"Failed to load model for checkpoint {checkpoint_step}. Skipping.")
        return

    dataset = torch.load(dataset_path) if dataset_path.endswith('.pt') else None
    if dataset is None:
        with open(dataset_path, 'r') as f:
            dataset = f.read()

    tokenized = chunk_and_tokenize(
        dataset,
        tokenizer,
        seq_len=128,
        batch_size=10
    )

    hidden_dim = model.gpt_neox.layers[layer_idx].mlp.dense_h_to_4h.out_features # extract from model
    activations = extract_activations(
        model=model,
        tokenized_data=tokenized,
        layer_idx=layer_idx,
        batch_size=10,
        num_samples=sae_config.batch_size,
        sequence_length=128
    )

    config = TrainConfig(
        d_in=hidden_dim,
        d_sae=int(hidden_dim * sae_config.expansion_factor),
        k=sae_config.k,
        learning_rate=1e-3,
        batch_size=sae_config.batch_size,
        l1_coefficient=sae_config.l1_coefficient,
        n_checkpoints=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    trainer = Trainer(config, activations, model)
    trainer.fit()

    layer_dir = f"./sae_checkpoints/layer_{layer_idx}"
    os.makedirs(layer_dir, exist_ok=True)
    sae_checkpoint_path = os.path.join(layer_dir, f"checkpoint_step_{checkpoint_step}.pt")

    torch.save(trainer.sae.state_dict(), sae_checkpoint_path)
    logging.info(
        f"SAE finished training for layer {layer_idx}, checkpoint {checkpoint_step}. "
        f"Saved state dict at: {sae_checkpoint_path}")
