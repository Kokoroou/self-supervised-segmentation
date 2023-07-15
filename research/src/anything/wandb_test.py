# I use wandb to log the hyper-parameters, training results, and model weights.
import random
from datetime import datetime
from pathlib import Path

import wandb


current_dir = Path(__file__).parent.resolve()

# Start a new wandb run to track this script
wandb.init(
    job_type="train",
    dir=current_dir.parent.parent.parent,
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 100,
    },
    project="semantic-segmentation",
    name="test_model_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    notes="test model",
    mode="online"
)

# simulate training
epochs = wandb.config.epochs
offset = random.random() / 5

for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [Optional] Finish the wandb run, necessary in notebooks
wandb.finish()
