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
