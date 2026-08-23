import os
import sys

import numpy as np

# uv run's process wrapping can leave stdout fully (not line-) buffered even
# in an interactive terminal, so ordinary print() calls sit invisible in the
# buffer for a long time instead of appearing per-step - force line buffering
# so training progress shows up immediately.
sys.stdout.reconfigure(line_buffering=True)

from nabla.tensor import Tensor
from nabla.backend import gpu_available, get_array_module, to_device
from nabla.checkpoint import save_checkpoint, load_checkpoint
from nabla.grad_mode import no_grad
from nabla.logging import History
from nabla.nn.unet import UNet
from nabla.nn.loss import BCEWithLogitsLoss, DiceLoss
from nabla.optim.adam import Adam
from nabla.optim.clip import clip_grad_norm_
from nabla.optim.scheduler import WarmupCosineLR

from data import split_by_patient, list_slice_files, get_batch, load_batch, filter_tumor_containing


def compute_loss(
    model: UNet, X: np.ndarray, y: np.ndarray, bce: BCEWithLogitsLoss, dice: DiceLoss, device: str
) -> tuple[Tensor, Tensor, Tensor]:
    """BCE(logits) + 5*Dice(sigmoid(logits)) - see docs/30-bce-with-logits.md.

    Pure Dice loss can get stuck predicting "all background" on BraTS's
    ~0.25%-positive tumor masks, since its gradient vanishes when the
    intersection stays near zero; BCE's per-pixel gradient doesn't have
    that problem and breaks the plateau, with Dice still driving overlap
    quality once training is past it. Weighting (1x BCE, 5x Dice) matches
    the PyTorch reference this pipeline was validated against.

    Returns (total, bce_loss, dice_loss) - the components are logged
    separately (see main()) because the *combined* number can look like
    it's improving purely from BCE learning the easy majority-background
    statistics while dice_loss - the actual tumor-overlap quality - stays
    stuck near 1.0, exactly the "predict all background" plateau BCE was
    added to escape. The combined loss alone can't distinguish those two
    situations.
    """
    logits = model(Tensor(X).to(device))
    targets = Tensor(y).to(device)
    bce_loss = bce(logits, targets)
    dice_loss = dice(logits.sigmoid(), targets)
    total = bce_loss + dice_loss * Tensor(np.array(5.0, dtype=np.float32)).to(device)
    return total, bce_loss, dice_loss

DATA_DIR = "D:\\projects\\brain-tumor-segmentation\\data\\processed_slices"
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), ".unet_checkpoint.pkl")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), ".unet_history.csv")

def main() -> None:
    # data first
    files = list_slice_files(DATA_DIR)
    train, val = split_by_patient(files, 0.1)
    rng = np.random.default_rng(0) # we need this to get batches

    print("Scanning val slices for tumor content (one-time, for a diagnostic metric)...")
    val_tumor = filter_tumor_containing(val)
    print(f"{len(val_tumor)} / {len(val)} val slices contain any tumor pixels")

    # hyperparams
    epochs = 8000
    # we need to use a low bs here bc my rtx 2070 super cant handle that much
    # original pytorch works with bs=16
    batch_size = 8
    eval_interval = 25
    # 240 is only evenly divisible by 16 (2**4, from the U-Net's 4 pooling
    # stages) at a downsample factor of 1 or 3 - 3 cuts H*W (and so Conv2D's
    # cost, which scales with it) by 9x, which is what makes training this
    # at all tractable on an 8GB GPU (see docs/28-unet.md / docs/29-no-grad.md)
    downsample = 3
    # each optimizer step averages gradients over this many batch_size=8
    # micro-batches before updating - a single batch of 8 gives a very
    # noisy gradient estimate (visible as val loss barely moving even as
    # train loss drops); accumulating grows the *effective* batch to
    # batch_size * accumulation_steps without needing more GPU memory,
    # since each micro-batch's forward/backward still only holds 8
    # samples' worth of activations at a time
    accumulation_steps = 8

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
    bce = BCEWithLogitsLoss()
    dice = DiceLoss()
    optimizer = Adam(model.parameters(), lr=3e-4)

    # logging
    history = History()

    # how many val micro-batches to average per eval - a single batch_size=8
    # batch is noisy given ~0.25% tumor pixels (an eval batch can easily
    # contain zero tumor-bearing slices), which makes it hard to tell a
    # real plateau apart from measurement noise. More batches only
    # stabilizes the number though - it doesn't fix what it's measuring:
    # DiceLoss scores a tumor-free slice ~1.0 (bad) the moment the
    # model's prediction has ANY diffuse non-zero mass anywhere, which is
    # normal sigmoid behavior, not a failure to detect anything. With
    # ~56% of val slices tumor-free, that drags the *aggregate* dice down
    # regardless of sample size, even when detection on the slices that
    # actually contain a tumor is good - see val_dice_tumor_only below,
    # which is the metric that actually answers "can this model find
    # tumors it's shown."
    val_eval_batches = 16
    val_tumor_only_batches = 4

    # training loop
    model.train()
    min_val_loss = float("inf")
    for epoch in range(epochs):
        model.zero_grad()
        accumulated_train_loss = 0.0
        accumulated_train_bce = 0.0
        accumulated_train_dice = 0.0
        for _ in range(accumulation_steps):
            X_train, y_train = get_batch(train, batch_size, rng, downsample=downsample, augment=True)
            train_loss, train_bce, train_dice = compute_loss(model, X_train, y_train, bce, dice, device)

            # scale the seed gradient by 1/accumulation_steps (equivalent to
            # averaging the losses first) so accumulated grads match what a
            # single batch_size*accumulation_steps batch would have produced
            xp = get_array_module(train_loss.data)
            train_loss.backward(xp.array(1.0 / accumulation_steps, dtype=train_loss.data.dtype))
            accumulated_train_loss += float(train_loss.data) / accumulation_steps
            accumulated_train_bce += float(train_bce.data) / accumulation_steps
            accumulated_train_dice += float(train_dice.data) / accumulation_steps
        # the PyTorch reference this pipeline was validated against clips
        # here too (max_norm=1.0) - without it, an occasional large batch
        # gradient can push a logit far enough that BCEWithLogitsLoss's
        # value (and its gradient) blows up to the hundreds or millions,
        # corrupting the weights it updates; grad_norm is logged so a
        # clip that's firing constantly (grad_norm >> 1.0 every step) is
        # visible as a stability signal, not just silently applied
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss = accumulated_train_loss
        train_bce = accumulated_train_bce
        train_dice = accumulated_train_dice

        if epoch % eval_interval == 0 or epoch == epochs - 1:
            model.eval()
            val_loss_sum = val_bce_sum = val_dice_sum = 0.0
            val_dice_tumor_sum = 0.0
            with no_grad():
                for _ in range(val_eval_batches):
                    X_val, y_val = get_batch(val, batch_size, rng, downsample=downsample)
                    vl, vb, vd = compute_loss(model, X_val, y_val, bce, dice, device)
                    val_loss_sum += float(vl.data)
                    val_bce_sum += float(vb.data)
                    val_dice_sum += float(vd.data)

                for _ in range(val_tumor_only_batches):
                    X_vt, y_vt = get_batch(val_tumor, batch_size, rng, downsample=downsample)
                    logits_vt = model(Tensor(X_vt).to(device))
                    dice_vt = dice(logits_vt.sigmoid(), Tensor(y_vt).to(device))
                    val_dice_tumor_sum += float(dice_vt.data)
            model.train()
            val_loss = val_loss_sum / val_eval_batches
            val_bce = val_bce_sum / val_eval_batches
            val_dice = val_dice_sum / val_eval_batches
            val_dice_tumor_only = val_dice_tumor_sum / val_tumor_only_batches

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                save_checkpoint(CHECKPOINT_PATH, model, optimizer, epoch=epoch, val_loss=val_loss)
                print(f"Checkpoint saved at epoch {epoch} with val loss {val_loss:.4f}")

            history.log(
                epoch,
                train_loss=train_loss,
                train_bce=train_bce,
                train_dice=train_dice,
                val_loss=val_loss,
                val_bce=val_bce,
                val_dice=val_dice,
                val_dice_tumor_only=val_dice_tumor_only,
                grad_norm=grad_norm,
                lr=optimizer.lr,
            )
            print(
                f"Epoch {epoch}: train loss = {train_loss:.4f} (bce={train_bce:.4f}, dice={train_dice:.4f}), "
                f"val loss = {val_loss:.4f} (bce={val_bce:.4f}, dice={val_dice:.4f}, "
                f"dice_tumor_only={val_dice_tumor_only:.4f}), grad_norm={grad_norm:.4f}"
            )

    history.to_csv(HISTORY_PATH)
    print(f"\nTraining complete. History saved to {HISTORY_PATH}.")


if __name__ == "__main__":
    main()