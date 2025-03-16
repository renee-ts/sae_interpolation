from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging


def load_pythia_checkpoint(checkpoint_step, model_name="EleutherAI/pythia-410m", device="cuda"):
    """
    Load a specific checkpoint step. There are 154 checkpoints:
        step{2^n} for n = {0, 1, ..., 10}
        step{1000m} for m = {1, 2, 3, ..., 143}.
    Source: https://huggingface.co/EleutherAI/pythia-410m-deduped/tree/main

    Args:
        checkpoint_step: the checkpoint step out of the 154 available.
        model_name: the model.
        device: default to gpu.

    Returns:
        model: the loaded model from the specified checkpoint
        tokenizer: the corresponding tokenizer
    """
    checkpoint_name = f"step{checkpoint_step}"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=checkpoint_name,
            torch_dtype=torch.bfloat16, # Maybe, if memory is an issue
            device_map={"": device}
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=checkpoint_name)
        logging.info(f"Loaded {model_name} at {checkpoint_name}")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load {model_name} at {checkpoint_name}: {e}")
        return None, None
    

def extract_activations(model, tokenized_data, layer_idx, num_samples=10000, sequence_length=128):
    """
    Extract activations from a specific transformer layer in the model.

    Args:
        model: the model.
        tokenized_data: the tokenized dataset.
        layer_idx: the index of the layer to extract activations from.
        num_samples: the number of activation samples to collect. Should be equal to max_samples in
            config.yaml.
        sequence_length: the length of token sequences to extract.

    Returns:
        A tensor of shape [num_samples, hidden_dim], where hidden_dim is the layer's dimension.
    """
    activations = []

    def hook(_, __, output):
        activations.append(output.detach().cpu())

    handle = None # TODO

    with torch.no_grad():
        raise NotImplementedError # TODO

    handle.remove()

    return torch.cat(activations, dim=0)[:num_samples]
