from collections import defaultdict
from dataclasses import asdict
from fnmatch import fnmatchcase
from glob import glob
from typing import Sized

from .pca_coder import PCA_Coder
import torch
import torch.distributed as dist
from datasets import Dataset as HfDataset
from natsort import natsorted
from safetensors.torch import load_model
from schedulefree import ScheduleFreeWrapper
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, get_linear_schedule_with_warmup

from sparsify import TrainConfig
from sparsify.data import MemmapDataset
from sparsify.utils import get_layer_list, resolve_widths, set_submodule

class LR_Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        dataset: HfDataset | MemmapDataset,
        model: PreTrainedModel,
    ):
        # Store the whole model, including any potential causal LM wrapper
        self.model = model

        if cfg.hookpoints:
            assert not cfg.layers, "Cannot specify both `hookpoints` and `layers`."

            # Replace wildcard patterns
            raw_hookpoints = []
            for name, _ in model.base_model.named_modules():
                if any(fnmatchcase(name, pat) for pat in cfg.hookpoints):
                    raw_hookpoints.append(name)

            # Natural sort to impose a consistent order
            cfg.hookpoints = natsorted(raw_hookpoints)
        else:
            # If no layers are specified, train on all of them
            if not cfg.layers:
                N = model.config.num_hidden_layers
                cfg.layers = list(range(0, N))

            # Now convert layers to hookpoints
            layers_name, _ = get_layer_list(model)
            cfg.hookpoints = [f"{layers_name}.{i}" for i in cfg.layers]

        cfg.hookpoints = cfg.hookpoints[:: cfg.layer_stride]

        self.cfg = cfg
        self.dataset = dataset
        self.distribute_modules()

        device = model.device
        input_widths = resolve_widths(model, cfg.hookpoints)
        unique_widths = set(input_widths.values())

        if cfg.distribute_modules and len(unique_widths) > 1:
            # dist.all_to_all requires tensors to have the same shape across ranks
            raise ValueError(
                f"All modules must output tensors of the same shape when using "
                f"`distribute_modules=True`, got {unique_widths}"
            )

        assert isinstance(dataset, Sized)
        num_batches = len(dataset) // cfg.batch_size

        self.pcas = {}
        for hook in self.local_hookpoints():
            for seed in cfg.init_seeds:
                torch.manual_seed(seed)

                # Add suffix to the name to disambiguate multiple seeds
                name = f"{hook}/seed{seed}" if len(cfg.init_seeds) > 1 else hook
                self.pcas[name] = PCA_Coder(
                    input_widths[hook],
                    cfg.pca,
                    device=device,
                    dtype=torch.float32
                )


        if self.cfg.save_best:
            self.best_loss = (
                {name: float("inf") for name in self.local_hookpoints()}
                if self.cfg.loss_fn == "fvu"
                else float("inf")
            )
        else:
            self.best_loss = None

    def load_state(self, path: str):
        """Load the trainer state from disk."""
        device = self.model.device

        # Load the train state first so we can print the step number
        train_state = torch.load(
            f"{path}/state.pt", map_location=device, weights_only=True
        )
        train_state["num_tokens_since_fired"] = {}

        for file in glob(f"{path}/rank_*_state.pt"):
            rank_train_state = torch.load(file, map_location=device, weights_only=True)
            train_state["num_tokens_since_fired"].update(
                rank_train_state["num_tokens_since_fired"]
            )

        self.global_step = train_state["global_step"]
        self.num_tokens_since_fired = {
            k: train_state["num_tokens_since_fired"][k] for k in self.local_hookpoints()
        }

        print(
            f"\033[92mResuming training at step {self.global_step} from '{path}'\033[0m"
        )

        for name, pca in self.pcas.items():
            assert isinstance(pca, PCA_Coder)
            pca.load_from_disk(f"{path}/{name}", device=device)

    def get_current_k(self) -> int:
        """Get the current k value based on a linear decay schedule."""
        if self.global_step >= self.cfg.k_decay_steps:
            return self.final_k

        progress = self.global_step / self.cfg.k_decay_steps
        return round(self.initial_k * (1 - progress) + self.final_k * progress)

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        # Make sure the model is frozen
        self.model.requires_grad_(False)

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized() and not self.cfg.distribute_modules

        wandb = None
        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb

                wandb.init(
                    name=self.cfg.run_name,
                    project="sparsify",
                    config=asdict(self.cfg),
                    save_code=True,
                )
            except (AttributeError, ImportError):
                print("Weights & Biases not available, skipping logging.")
                print("Run `pip install -U wandb` if you want to use it.")
                self.cfg.log_to_wandb = False

        num_model_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of model parameters: {num_model_params:_}")

        num_batches = len(self.dataset) // self.cfg.batch_size
        if self.global_step > 0:
            assert hasattr(self.dataset, "select"), "Dataset must implement `select`"

            n = self.global_step * self.cfg.batch_size
            ds = self.dataset.select(range(n, len(self.dataset)))  # type: ignore
        else:
            ds = self.dataset

        device = self.model.device
        dl = DataLoader(
            ds,  # type: ignore
            batch_size=self.cfg.batch_size,
            # NOTE: We do not shuffle here for reproducibility; the dataset should
            # be shuffled before passing it to the trainer.
            shuffle=False,
        )
        avg_fvu = defaultdict(float)
        avg_multi_topk_fvu = defaultdict(float)
        avg_losses = (
            {name: float("inf") for name in self.local_hookpoints()}
        )
        num_tokens_in_step = 0

        name_to_module = {
            name: self.model.base_model.get_submodule(name)
            for name in self.cfg.hookpoints
        }
        maybe_wrapped: dict[str, DDP] | dict[str, PCA_Coder] = {}
        module_to_name = {v: k for k, v in name_to_module.items()}

        def hook(module: nn.Module, inputs, outputs):
            aux_out = None

            # Maybe unpack tuple inputs and outputs
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            if isinstance(outputs, tuple):
                outputs, *aux_out = outputs

            # Name may optionally contain a suffix of the form /seedN where N is an
            # integer. We only care about the part before the slash.
            name, _, _ = module_to_name[module].partition("/")

            # Remember the original output shape since we'll need it for e2e training
            out_shape = outputs.shape

            # Scatter and gather the hidden states across ranks if necessary
            if self.cfg.distribute_modules:
                world_outputs = outputs.new_empty(
                    outputs.shape[0] * dist.get_world_size(), *outputs.shape[1:]
                )
                dist.all_gather_into_tensor(world_outputs, outputs)
                outputs = world_outputs

                if name not in self.module_plan[dist.get_rank()]:
                    return

            # Flatten the batch and sequence dimensions
            outputs = outputs.flatten(0, 1)
            # Get the mean of the activations
            pca_coder = maybe_wrapped[name]
            pca_coder.fit(outputs)

            # low_rank_out = self.pcas[name]["U"] @ torch.diag(self.pcas[name]["S"]) @ self.pcas[name]["V"].T + self.pcas[name]["mean"]
            low_rank_out = pca_coder.U @ torch.diag(pca_coder.S) @ pca_coder.Vh + pca_coder.mean
            assert low_rank_out.shape == outputs.shape, (
                low_rank_out.shape, outputs.shape
            )

            # Calculate FVU loss
            y = outputs
            res = y - low_rank_out
            l2_loss = torch.linalg.norm(res, ord=2, dim=1).mean()
            fvu_loss = l2_loss / torch.linalg.norm(y, ord=2, dim=1).mean()
            avg_fvu[name] += float(self.maybe_all_reduce(fvu_loss.detach(), op="mean"))


        # k = self.get_current_k()
        # for name, sae in self.saes.items():
        #     sae.cfg.k = k
        # TODO: May want to implement rank decay later. That is, we start at a higher rank and then gradually get towards the final rank--might help when doing EMA.

        for batch in dl:
            x = batch["input_ids"].to(device)

            if not maybe_wrapped:
                # Wrap the SAEs with Distributed Data Parallel. We have to do this
                # after we set the decoder bias, otherwise DDP will not register
                # gradients flowing to the bias after the first step.
                maybe_wrapped = (
                    {
                        name: DDP(pca, device_ids=[dist.get_rank()])
                        for name, pca in self.pcas.items()
                    }
                    if ddp
                    else self.pcas
                )

            # Bookkeeping for dead feature detection
            N = x.numel()
            num_tokens_in_step += N

            # Forward pass on the model to get the next batch of activations
            handles = [
                mod.register_forward_hook(hook) for mod in name_to_module.values()
            ]

            try:
                self.model(x)
                avg_losses = avg_fvu
            finally:
                for handle in handles:
                    handle.remove()

            step = self.global_step + 1
            if (
                self.cfg.log_to_wandb
                and (step + 1) % self.cfg.wandb_log_frequency == 0
            ):
                info = {}

                for name in self.pcas:
                    mask = (
                        self.num_tokens_since_fired[name]
                        > self.cfg.dead_feature_threshold
                    )

                    ratio = mask.mean(dtype=torch.float32).item()
                    info.update({f"dead_pct/{name}": ratio})
                    if self.cfg.loss_fn == "fvu":
                        info[f"fvu/{name}"] = avg_fvu[name]

                avg_fvu.clear()
                avg_multi_topk_fvu.clear()

                if self.cfg.distribute_modules:
                    outputs = [{} for _ in range(dist.get_world_size())]
                    dist.gather_object(info, outputs if rank_zero else None)
                    info.update({k: v for out in outputs for k, v in out.items()})

                if rank_zero:
                    info["k"] = self.rank

                    if wandb is not None:
                        wandb.log(info, step=step)

            if (step + 1) % self.cfg.save_every == 0:
                self.save()

                if self.cfg.save_best:
                    self.save_best(avg_losses)

            self.global_step += 1

        self.save()
        if self.cfg.save_best:
            self.save_best(avg_losses)


    def local_hookpoints(self) -> list[str]:
        return (
            self.module_plan[dist.get_rank()]
            if self.module_plan
            else self.cfg.hookpoints
        )

    def maybe_all_cat(self, x: Tensor) -> Tensor:
        """Concatenate a tensor across all processes."""
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x)
        return buffer

    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"Unknown reduction op '{op}'")

        return x

    def distribute_modules(self):
        """Prepare a plan for distributing modules across ranks."""
        if not self.cfg.distribute_modules:
            self.module_plan = []
            print(f"Training on modules: {self.cfg.hookpoints}")
            return

        layers_per_rank, rem = divmod(len(self.cfg.hookpoints), dist.get_world_size())
        assert rem == 0, "Number of modules must be divisible by world size"

        # Each rank gets a subset of the layers
        self.module_plan = [
            self.cfg.hookpoints[start : start + layers_per_rank]
            for start in range(0, len(self.cfg.hookpoints), layers_per_rank)
        ]
        for rank, modules in enumerate(self.module_plan):
            print(f"Rank {rank} modules: {modules}")

    def save_best(self, avg_loss: float | dict[str, float]):
        """Save individual sparse coders to disk if they have the lowest loss."""
        base_path = f'{self.cfg.save_dir}/{self.cfg.run_name or "unnamed"}/best'
        if type(avg_loss) is float:
            if avg_loss < self.best_loss:  # type: ignore
                self.best_loss = avg_loss  # type: ignore
                self.save(base_path)
        else:
            for name in self.pcas:
                if avg_loss[name] < self.best_loss[name]:  # type: ignore
                    self.best_loss[name] = avg_loss[name]  # type: ignore
                    path = f"{base_path}/{name}"
                    self.save(path, {name: self.pcas[name]})

    def save(self, path: str | None = None, pcas: dict[str, PCA_Coder] | None = None):
        """Save the SAEs to disk."""
        if path is None:
            path = f'{self.cfg.save_dir}/{self.cfg.run_name or "unnamed"}'
        if pcas is None:
            pcas = self.pcas

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        if rank_zero or self.cfg.distribute_modules:
            print("Saving checkpoint")


            for name, pca in pcas.items():
                assert isinstance(pca, PCA_Coder)
                pca.save_to_disk(f"{path}/{name}")

            rank = 0 if rank_zero else dist.get_rank()
            torch.save(
                {"num_tokens_since_fired": self.num_tokens_since_fired},
                f"{path}/rank_{rank}_state.pt",
            )

        if rank_zero:
            torch.save({"global_step": self.global_step}, f"{path}/state.pt")

            self.cfg.save_json(f"{path}/config.json")

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()


# Support old name for compatibility
SaeTrainer = Trainer