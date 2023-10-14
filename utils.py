import torch


def get_batch(data, batch_size, device, device_type):
    """
    sample a random batch from the train or val data
    """
    # data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - 1, (batch_size,))

    x = torch.tensor([data[i][1] for i in ix], dtype=torch.long)
    y = torch.tensor([data[i][0] for i in ix], dtype=torch.long)

    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)

    return x, y


def save_model(model, optimizer, epoch, accuracy, ddp, log_wandb=False):
    """save model to directory and optionally to wandb"""

    import datetime

    # get today's date
    model_suffix = datetime.today().strftime("%Y-%m-%d")

    output_model = model.module.state_dict() if ddp else model.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": output_model,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_accuracy": accuracy["train"],
            "val_accuracy": accuracy["val"],
        },
        f"model_{model_suffix}.pt",
    )

    if log_wandb:
        import wandb

        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(f"model_{model_suffix}.pt")
        wandb.log_artifact(artifact)


@torch.no_grad()
def estimate_loss(model, eval_iters):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            if k % 10 == 0:
                print(f"evaluating {split} loss: {k}/{eval_iters}")

            X, Y = get_batch(split)
            logits, loss = model(X, labels=Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out


@torch.no_grad()
def estimate_accuracy(model, eval_iters):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        correct = torch.zeros(eval_iters)

        for k in range(eval_iters):
            if k % 10 == 0:
                print(f"evaluating {split} accuracy: {k}/{eval_iters}")

            X, Y = get_batch(split)
            logits, loss = model(X, labels=Y)

            correct[k] = (logits.argmax(dim=1) == Y).float().mean().item()

        out[split] = correct.mean()

    model.train()
    return out
