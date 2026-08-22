import os
import sys

import numpy as np

# uv run's process wrapping can leave stdout fully (not line-) buffered even
# in an interactive terminal, so ordinary print() calls sit invisible in the
# buffer for a long time instead of appearing per-step - force line buffering
# so training progress shows up immediately.
sys.stdout.reconfigure(line_buffering=True)

from nabla.tensor import Tensor
from nabla.backend import gpu_available, to_device
from nabla.checkpoint import save_checkpoint, load_checkpoint
from nabla.grad_mode import no_grad
from nabla.logging import History
from nabla.nn.unet import UNet
from nabla.nn.loss import DiceLoss
from nabla.optim.adam import Adam
from nabla.optim.scheduler import WarmupCosineLR

from data import split_by_patient, list_slice_files, get_batch, load_batch


def compute_loss(
    model: UNet, X: np.ndarray, y: np.ndarray, criterion, device: str
) -> Tensor:
    logits = model(Tensor(X).to(device))  # (batch, block_size, vocab_size)
    return criterion(logits, Tensor(y).to(device))

DATA_DIR = "D:\\projects\\brain-tumor-segmentation\\data\\processed_slices"
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), ".unet_checkpoint.pkl")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), ".unet_history.csv")

def main() -> None:
    # data first
    files = list_slice_files(DATA_DIR)
    train, val = split_by_patient(files, 0.1)
    rng = np.random.default_rng(0) # we need this to get batches

    # hyperparams
    epochs = 8000
    # we need to use a low bs here bc my rtx 2070 super cant handle that much
    # original pytorch works with bs=16 
    batch_size = 8
    eval_interval = 25

    # model setup
    device = "cuda" if gpu_available() else "cpu"
    print(f"Training on {device}")

    features = (64, 128, 256, 512)
    # we use FLAIR + T1ce slices as input
    model = UNet(
        in_channels=2,
        out_channels=1,
        features=features
    )
    model.to(device)

    # optim setup
    criterion = DiceLoss()
    optimizer = Adam(model.parameters(), lr=3e-4)

    # logging
    history = History()

    # training loop
    model.train()
    min_val_loss = float("inf")
    for epoch in range(epochs):
        X_train, y_train = get_batch(train, batch_size, rng)
        
        train_loss = compute_loss(model, X_train, y_train, criterion, device)
        
        model.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch % eval_interval == 0 or epoch == epochs - 1:
            model.eval()
            with no_grad():
                X_val, y_val = get_batch(val, batch_size, rng)
                val_loss = compute_loss(model, X_val, y_val, criterion, device)
            model.train()

            if val_loss.data < min_val_loss:
                min_val_loss = val_loss.data
                save_checkpoint(CHECKPOINT_PATH, model, optimizer, epoch=epoch, val_loss=val_loss.data)
                print(f"Checkpoint saved at epoch {epoch} with val loss {val_loss.data:.4f}")

            history.log(epoch, train_loss=float(train_loss.data), val_loss=float(val_loss.data), lr=optimizer.lr)
            print(f"Epoch {epoch}: train loss = {train_loss.data:.4f}, val loss = {val_loss.data:.4f}")

    history.to_csv(HISTORY_PATH)
    print(f"\nTraining complete. History saved to {HISTORY_PATH}.")


if __name__ == "__main__":
    main()