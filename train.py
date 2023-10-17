""" Train a transformer model on a classification task. """
import os
import json
import yaml
import wandb
import torch
import logging
import requests
import torch.distributed as dist
from dataloader import load_data
from model import Classifier, ModelConfig
from transformers import GPT2ForSequenceClassification
from utils import get_batch, save_model, estimate_loss, estimate_accuracy


logging.basicConfig(
    filename="log.log", level=logging.INFO, format="%(asctime)s %(message)s"
)

with open("creds/creds.json", "r") as f:
    creds = json.load(f)

user = creds["pushover"]["user"]
token = creds["pushover"]["token"]

# load config from config.yml file
with open("config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# model parameters
pretrained = config["pretrained"]
pretrained_model_name = config["model_name"]
n_layer = config["n_layer"]
dropout = config["dropout"]
n_head = config["n_head"]
head_size = config["head_size"]
n_embed = head_size * n_head
dropout = config["dropout"]


# data parameters
block_size = config["block_size"]
batch_size = config["batch_size"]
vocab_size = config["vocab_size"]
output_size = config["output_size"]

# training parameters
max_iters = config["max_iters"]
max_interval = config["max_interval"]
eval_iters = config["eval_iters"]
learning_rate = config["learning_rate"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# toggles for training and logging
log_wandb = config["log_wandb"]
log_pushover = config["log_pushover"]

# ddp parameters
ddp = config["ddp"]
if ddp:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    device_type = "cuda"
    master_device = rank == 0
    dist.init_process_group(backend="nccl")
else:
    device = torch.device(device)
    device_type = device.type
    master_device = True


if master_device and log_wandb:
    wandb.init(
        project=config["project"],
        config=config,
        save_code=True,
    )


model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    head_size=head_size,
    n_embed=n_embed,
    block_size=block_size,
    vocab_size=vocab_size,
    dropout=dropout,
    output_size=output_size,
)

config = ModelConfig(**model_args)

for split in ["train", "val"]:
    dataset = load_data(split)

    if split == "train":
        train_data = dataset
    else:
        val_data = dataset


if pretrained:
    model = model = GPT2ForSequenceClassification.from_pretrained(
        pretrained_model_name, num_labels=output_size
    )
    model.to(device)
    model.config.pad_token_id = model.config.eos_token_id
else:
    model = Classifier(config, device).to(device)

if ddp:  # wrap the model in DDP
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if master_device:
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

if master_device and log_pushover:
    data = {
        "token": token,
        "user": user,
        "sound": "echo",
        "message": "training started ...",
    }
    r = requests.post("https://api.pushover.net/1/messages.json", data=data)


for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_iters == 0 and master_device:
        print("estimate accuracy...")
        losses = estimate_loss(model, eval_iters)
        accuracy = estimate_accuracy(model, eval_iters)

        text = [
            f"step {iter} - {device}: train accuracy {accuracy['train']:.4f}, val accuracy {accuracy['val']:.4f}, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}",
        ]

        logging.info(text)

        if log_wandb:
            wandb.log(
                {
                    "iter": iter,
                    "train_accuracy": accuracy["train"],
                    "val_accuracy": accuracy["val"],
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                }
            )

        if log_pushover:
            data = {
                "token": token,
                "user": user,
                "sound": "echo",
                "message": text + "\n",
            }
            r = requests.post("https://api.pushover.net/1/messages.json", data=data)

    # sample a batch of data
    X, Y = get_batch("train")
    logits, loss = model(X, labels=Y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % 10 == 0 and master_device:
        logging.info(f"step {iter} - {device}")

    if iter % 1000 == 0 and master_device:
        save_model(model, optimizer, iter, accuracy, ddp)
        print(f"model saved at step {iter} - {device}")


if ddp:
    dist.destroy_process_group()
