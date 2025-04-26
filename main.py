import argparse
import logging
import os
from typing import Dict, Any
import yaml
import torch
import gc

from training.train_sae import train_sae
from utils.logging_utils import setup_logging

CHECKPOINTS = sorted([i * 1000 for i in range(144)] + [2 ** p for p in range(10)])

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

    for step in assigned_checkpoints:
        checkpoint_output_dir = os.path.join(args.output_dir, f"checkpoint_{step}")
        os.makedirs(checkpoint_output_dir, exist_ok=True)
        sae_config = sae_base_config.copy()
        sae_config["output_dir"] = checkpoint_output_dir
        logging.info(f"Training SAE for checkpoint {step}")
        try:
            train_sae(step, sae_config)
            logging.info(f"Completed checkpoint {step}")
        except Exception as e:
            logging.error(f"Error on checkpoint {step}: {str(e)}")

if __name__ == "__main__":
    main()