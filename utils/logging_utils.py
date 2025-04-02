import logging
import os


def setup_logging(checkpoint_step: int, layer_idx: int):
    """
    Sets up logging configuration.
    """
    log_dir = f"./logs/step_{checkpoint_step}"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"layer_{layer_idx}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
