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
    

def extract_activations(model, tokenized_data, layer_idx, batch_size = 10, num_samples=10000, sequence_length=128):
    """
    Extract activations from a specific transformer layer in the model.

    Args:
        model: the model.
        tokenized_data: the tokenized dataset.
        layer_idx: the index of the layer to extract activations from.
        batch_size: the number of sequences to process at every sample iteration.
        num_samples: the number of activation samples to collect. Should be equal to max_samples in
            config.yaml.
        sequence_length: the length of token sequences to extract.

    Returns:
        A tensor of shape [num_samples, hidden_dim], where hidden_dim is the layer's dimension.
    """
    activations = []
    
    #hook() is called every time layer `layer_idx` is passed, we append layer activations to `activations`
    def hook(_, __, output):
        if isinstance(output, tuple):  
            output = output[0]
        activations.append(output.detach().cpu())

    #configure forward hook at layer `layer_idx`
    handle = model.gpt_neox.layers[layer_idx].register_forward_hook(hook)

    with torch.no_grad():
        #sample dataset `num_sample` times
        for i in range(0, num_samples):
            start, end = i * batch_size, (i + 1) * batch_size
            #get `batch_size` number of `sequence_length` tokens from tokenized sequence
            batch = {'input_ids': tokenized_data['input_ids'][start:end, 0:sequence_length], 'attention_mask' : tokenized_data['attention_mask'][start:end, 0:sequence_length]}
            #run forward-pass
            _ = model(**batch)

    handle.remove()
    
    #stack activation lists of `num_samples` tensors into one tensor, and reshape the tensor
    return torch.stack(activations, dim=0)[:num_samples].reshape(num_samples*sequence_length*batch_size, 768)
