import argparse
import logging
from dotenv import load_dotenv
from typing import Dict, Any
import os
import yaml
import torch
import gc
import glob

from training.train_sae import train_sae
from hf_upload import already_uploaded
load_dotenv()

CHECKPOINTS = sorted([2 ** p for p in range(10)] + [n * 1000 for n in range(44)] + [m * 1000 for m in range(45, 144, 2)])
LAYERS = [0, 6, 11]


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    torch.cuda.empty_cache()
    gc.collect()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="should just be the config.yaml")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--total_tasks", type=int, required=True, help="How many GPUs are being utilized")
    parser.add_argument("--task_id", type=int, required=True)
    args = parser.parse_args()

    checkpoints = CHECKPOINTS
    slice_size = (len(checkpoints) + args.total_tasks - 1) // args.total_tasks
    start = args.task_id * slice_size
    end = min(start + slice_size, len(checkpoints))
    assigned_checkpoints = checkpoints[start:end]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.output_dir}/train_task_{args.task_id}.log"),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Task {args.task_id} handling {len(assigned_checkpoints)} checkpoints")

    full_config = load_config(args.config)
    sae_base_config = full_config["sae"]

    hf_token = os.getenv("ORG_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment.")

    SCRATCH_DIR = "/scratch/ssd004/scratch/charless/sae_checkpoints"
    for step in assigned_checkpoints:

        if already_uploaded(step, hf_token):
            logging.info(f"Checkpoint {step} already uploaded — skipping.")
            continue

        sae_config = sae_base_config.copy()
        sae_config["output_dir"] = SCRATCH_DIR
        logging.info(f"Training SAE for checkpoint {step}")
        try:
            save_dir = train_sae(step, sae_config, layers=LAYERS)
            best_files = glob.glob(os.path.join(save_dir, "best", "**", "sae.safetensors"), recursive=True)
            if not best_files:
                print(f"No best SAE files found in {save_dir} — skipping upload.")

            logging.info(f"Completed checkpoint {step}")
            print("saved")
        except Exception as e:
            logging.error(f"Error on checkpoint {step}: {str(e)}")

if __name__ == "__main__":
    main()