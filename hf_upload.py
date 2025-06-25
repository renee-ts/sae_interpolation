import os
import shutil
from huggingface_hub import HfApi

HF_REPO_ID = "sae-ml-drp/sae-ml-drp"
EXCLUDE_EXTENSIONS = {".pt", ".json"}

def already_uploaded(step, hf_token):
    api = HfApi(token=hf_token)
    repo_files = api.list_repo_files("sae-ml-drp/sae-ml-drp", token=hf_token)
    return any(f.startswith(f"step-{step}/") and f.endswith(".safetensors") for f in repo_files)

def upload_folder_to_hf(local_folder, repo_id, repo_path_prefix="", hf_token=None):
    if not hf_token:
        raise ValueError("No HF token.")
    api = HfApi(token=hf_token)
    
    for root, _, files in os.walk(local_folder):
        for file in files:

            if any(file.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
                continue
            if not file.endswith(".safetensors"):
                continue
            
            local_path = os.path.join(root, file)

            if '/best/' not in local_path.replace('\\', '/'):
                print(f"Skipping non-best file: {local_path}")
                continue
                
            relative_path = os.path.relpath(local_path, local_folder)
            repo_path = os.path.join(repo_path_prefix, relative_path).replace("\\", "/")
            
            print(f"Uploading {local_path} to {repo_path}.")
            
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model",
                    token=hf_token,
                )
                try:
                    os.remove(local_path)
                    print(f"Path deleted.")
                except Exception as e:
                    print(f"Failed to delete {local_path}: {e}")

            except Exception as e:
                print(f"Failed to upload {local_path}: {e}.")

scratch_dir = "/scratch/ssd004/scratch/charless/sae_checkpoints" # hardcode is fine

hf_token = os.getenv("ORG_TOKEN")

def upload_existing():
    for checkpoint_dir in os.listdir(scratch_dir):
        if checkpoint_dir.startswith("step-"):
            _ = checkpoint_dir.replace("step-", "")
            local_folder = os.path.join(scratch_dir, checkpoint_dir)
            
            print(f"Uploading {checkpoint_dir}...")
            upload_folder_to_hf(
                local_folder=local_folder,
                repo_id="sae-ml-drp/sae-ml-drp", 
                repo_path_prefix=checkpoint_dir,
                hf_token=hf_token
            )

            shutil.rmtree(local_folder)
            print(f"Deleted {local_folder}")


if __name__ == "__main__":
    upload_existing()
