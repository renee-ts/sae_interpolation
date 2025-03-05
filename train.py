import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

model_name = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# RedPajama sample for demo (should use correct path for Vector's machines)
dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
tokenized = chunk_and_tokenize(dataset, tokenizer)

trained_model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

trained_state_dict = trained_model.state_dict()
random_state_dict = {}
for key, param in trained_state_dict.items():
    mean = param.mean().item()
    std = param.std().item()
    random_param = torch.normal(mean, std, size=param.shape, device="cuda")
    random_state_dict[key] = random_param
random_model = AutoModelForCausalLM.from_config(trained_model.config).to("cuda")
random_model.load_state_dict(random_state_dict)

excl_emb_state_dict = random_state_dict.copy()
excl_emb_state_dict["model.embed_tokens.weight"] = trained_state_dict["model.embed_tokens.weight"]
excl_emb_model = AutoModelForCausalLM.from_config(trained_model.config).to("cuda")
excl_emb_model.load_state_dict(excl_emb_state_dict)


class ControlModel(AutoModelForCausalLM): # gaussian noise embeddings
    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = torch.randn(
                input_ids.shape[0],
                input_ids.shape[1],
                self.config.hidden_size,
                device=input_ids.device
            )
        return super().forward(inputs_embeds=inputs_embeds, **kwargs)

control_model = ControlModel.from_pretrained(model_name).to("cuda")

# hookpoints every 4th layer (adjustable for different models)
num_layers = trained_model.config.num_hidden_layers
hookpoints = [f"model.layers.{i}" for i in range(0, num_layers, 4)]
sae_config = SaeConfig(
    hookpoints=hookpoints,
    expansion_factor=64,  # from the sae paper
    k=32,  # can change
)

train_config = TrainConfig(
    sae_config,
    batch_size=8,  # randomly chosen as of now we can change it later
)

# variants to train SAEs on
variants = {
    "trained": trained_model,
    "random_incl_emb": random_model,
    "random_excl_emb": excl_emb_model,
    "control": control_model
}

for variant_name, model in variants.items():
    print(f"Training SAE for {variant_name}...")
    trainer = Trainer(train_config, tokenized, model)
    trainer.fit()
