import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files, login, HfApi
import safetensors.torch
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import gc
import psutil
import sys

REPO_ID = "sae-ml-drp/sae-ml-drp"
LAYER_TO_ANALYZE=6 # change between [0,6,11]
CHECKPOINTS = sorted([2 ** p for p in range(10)] + [n * 1000 for n in range(44)] + [m * 1000 for m in range(45, 144, 2)])

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_memory_usage(label=""):
    """Print current memory usage"""
    memory_mb = get_memory_usage()
    print(f"Memory usage {label}: {memory_mb:.1f} MB")

def setup_authentication():
    """
    Set up HuggingFace authentication using token from environment variable or prompt user.
    """
    token = os.environ.get("HF_ACCESS_TOKEN")
    
    if not token:
        print("No HuggingFace token found in environment variables.")
        print("Please set one of the following environment variables:")
        print("  - HF_ACCESS_TOKEN")
        print("  - HUGGING_FACE_HUB_TOKEN")
        print("\nOr you can login interactively:")
        try:
            login()
            print("Login successful!")
            return True
        except Exception as e:
            print(f"Login failed: {e}")
            return False
    else:
        print("Using HuggingFace token from environment variable.")
        return True

def verify_authentication():
    """
    Verify that authentication is working by testing access to the repository.
    """
    try:
        api = HfApi()
        # Try to get repository info
        repo_info = api.repo_info(REPO_ID, repo_type="model")
        print(f"✓ Successfully authenticated! Repository: {repo_info.id}")
        print(f"  - Repository type: {getattr(repo_info, 'type', 'model')}")
        print(f"  - Private: {getattr(repo_info, 'private', 'unknown')}")
        return True
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return False

def load_weights(repo_id, checkpoint, layer):
    """
    Downloads (if not cached) and loads the encoder.weight tensor for a given checkpoint and layer.
    Returns the weight tensor and the filepath for cleanup.
    """
    try:
        # token from environment variable
        token = os.environ.get("HF_ACCESS_TOKEN")
        
        filename = f"step-{checkpoint}/step-{checkpoint}/best/layers.{layer}/layers.{layer}/sae.safetensors"
        print(f"Attempting to load weights from: {filename}")
        filepath = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token
        )
        print(f"Successfully downloaded to: {filepath}")
        
        tensors = safetensors.torch.load_file(filepath)
        
        if 'encoder.weight' not in tensors:
            # Available keys in safetensors file: ['W_dec', 'b_dec', 'encoder.bias', 'encoder.weight']
            possible_keys = [k for k in tensors.keys() if 'enc' in k.lower() or 'encoder' in k.lower()]
            if possible_keys:
                print(f"Found similar keys: {possible_keys}")
            raise KeyError(f"encoder.weight not found in safetensors file for checkpoint {checkpoint}, layer {layer}")
        
        weight = tensors['encoder.weight'].to(torch.float32).cpu()
        del tensors
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return weight, filepath
    except Exception as e:
        print(f"Error loading weights from {filename}: {e}")
        return None, None
    
def align_weights(prev_aligned_w, current_w):
    """
    Aligns the rows (latent features) of current_w to prev_aligned_w using greedy matching
    on cosine similarity. Much more memory efficient than Hungarian algorithm.

    Args:
        prev_aligned_w (Tensor): The weight matrix from the previous step, already aligned.
                                 Shape: [n_latents, d_model]
        current_w (Tensor): The weight matrix from the current step.
                            Shape: [n_latents, d_model]

    Returns:
        Tensor: The aligned current_w matrix.
    """
    # Use float32 for memory efficiency
    prev_aligned_w = prev_aligned_w.float()
    current_w = current_w.float()
    
    print("Normalizing weights...")
    prev_norm = F.normalize(prev_aligned_w, p=2, dim=1)
    current_norm = F.normalize(current_w, p=2, dim=1)
    
    print("Computing cost matrix...")
    similarity_matrix = torch.matmul(prev_norm, current_norm.T)
    
    print("Finding optimal assignment...")
    n_latents = similarity_matrix.shape[0]
    used_indices = set()
    permutation = []
    
    for i in range(n_latents):
        row_similarities = similarity_matrix[i].clone()
        for used_idx in used_indices:
            row_similarities[used_idx] = -1.0  # make used indices have lowest similarity
        
        best_idx = torch.argmax(row_similarities).item()
        permutation.append(best_idx)
        used_indices.add(best_idx)
    
    aligned_w = current_w[permutation, :]
    
    del prev_norm, current_norm, similarity_matrix, row_similarities
    gc.collect()
    
    return aligned_w

def cleanup_file(filepath):
    """
    Clean up downloaded file to free disk space.
    """
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            print(f"Cleaned up: {filepath}")
    except Exception as e:
        print(f"Warning: Could not clean up {filepath}: {e}")

def process_checkpoints_in_batches(checkpoint_names, batch_size=2):
    """
    Process checkpoints in smaller batches to manage memory better.
    """
    print(f"Processing {len(checkpoint_names)} checkpoints in batches of {batch_size}")
    
    # Load initial weights
    print(f"Loading initial reference weights from checkpoint step-{checkpoint_names[0]}...")
    w_initial, initial_filepath = load_weights(REPO_ID, checkpoint_names[0], LAYER_TO_ANALYZE)
    
    if w_initial is None:
        print(f"Fatal: Could not load initial checkpoint step-{checkpoint_names[0]}. Aborting.")
        return None
    
    print_memory_usage("after loading initial weights")
    
    # Store flattened initial weights for PCA
    flattened_vectors = [w_initial.flatten().numpy()]
    last_aligned_w = w_initial
    
    cleanup_file(initial_filepath)
    del w_initial
    gc.collect()
    
    for i in range(1, len(checkpoint_names), batch_size):
        batch = checkpoint_names[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}: checkpoints {i+1}-{min(i+batch_size, len(checkpoint_names))}")
        
        for step in tqdm(batch, desc=f"Batch {i//batch_size + 1}"):
            w_current, filepath = load_weights(REPO_ID, step, LAYER_TO_ANALYZE)
            if w_current is None:
                continue
            
            # align the current weights to the previously aligned weights
            w_aligned = align_weights(last_aligned_w, w_current)
            
            flattened_vectors.append(w_aligned.flatten().numpy())
            
            last_aligned_w = w_aligned
            cleanup_file(filepath)
            del w_current, w_aligned
            gc.collect()
            
            print_memory_usage(f"after processing {step}")
        
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print_memory_usage(f"after batch {i//batch_size + 1}")
    
    return flattened_vectors

def main():
    print(f"Starting analysis for layer: {LAYER_TO_ANALYZE}")
    print_memory_usage("at start")
    
    print("Setting up HuggingFace authentication...")
    if not setup_authentication():
        print("Failed to set up authentication. Exiting.")
        return
    
    print("Verifying authentication...")
    if not verify_authentication():
        print("Authentication verification failed. Exiting.")
        return
        
    checkpoint_names = [f"{step}" for step in CHECKPOINTS]
    
    flattened_vectors = process_checkpoints_in_batches(checkpoint_names, batch_size=1)
    
    if flattened_vectors is None or len(flattened_vectors) < 2:
        print("Not enough data to run PCA. Need at least 2 valid CHECKPOINTS.")
        return

    print(f"Successfully processed {len(flattened_vectors)} CHECKPOINTS.")
    print_memory_usage("before creating PCA dataset")

    # 2D dataset for PCA
    print("Creating dataset for PCA...")
    X = np.stack(flattened_vectors, axis=0)
    
    del flattened_vectors
    gc.collect()
    print_memory_usage("after creating PCA dataset")

    print("Running PCA...")
    # analyze the top 10 principal components
    n_components = min(10, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    
    pca.fit(X)

    # display results
    print("\n--- PCA Results ---")
    print(f"Analyzed {X.shape[0]} CHECKPOINTS for layer {LAYER_TO_ANALYZE}.")
    print("Explained Variance Ratio per Component:")
    for i, variance in enumerate(pca.explained_variance_ratio_):
        print(f"  PC-{i+1}: {variance:.4f} ({variance*100:.2f}%)")
        
    print(f"\nTotal variance explained by top {n_components} components: {sum(pca.explained_variance_ratio_):.4f} ({sum(pca.explained_variance_ratio_)*100:.2f}%)")
    
    del X, pca
    gc.collect()
    print_memory_usage("at end")

if __name__ == "__main__":
    main()
    