"""Save/load training state to disk, so a long run can resume after a
crash (or just be paused) instead of starting over from scratch.

Built on Module.state_dict()/load_state_dict() and
Optimizer.state_dict()/load_state_dict() rather than reaching into a
model's internals directly - checkpoint.py itself doesn't know or care
whether the optimizer is Adam or SGD, or what layers the model has.
"""

from __future__ import annotations

import pickle
from typing import Any

from nabla.nn.module import Module
from nabla.optim.optimizer import Optimizer


def save_checkpoint(path: str, model: Module, optimizer: Optimizer | None = None, **metadata: Any) -> None:
    """Save model (and optionally optimizer) state to `path`.

    Args:
        path: File path to write to.
        model: The model whose parameters get saved.
        optimizer: If given, its internal state (e.g. Adam's momentum
            buffers) is saved too, so resuming training doesn't restart
            momentum from zero.
        **metadata: Arbitrary extra values to save alongside the state
            (e.g. `epoch=12, step=4000, best_val_loss=1.23`) - returned
            back as a dict by `load_checkpoint`.

    Every array is moved to the CPU before saving, regardless of which
    device the model/optimizer currently live on - so a checkpoint saved
    from a GPU run can still be loaded (and inspected) on a CPU-only
    machine, and vice versa.
    """
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "metadata": metadata,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: str, model: Module, optimizer: Optimizer | None = None) -> dict[str, Any]:
    """Load a checkpoint saved by `save_checkpoint` back into `model`
    (and `optimizer`, if given and the checkpoint has optimizer state).

    Loaded values land on whatever device `model`'s (and `optimizer`'s)
    parameters are *currently* on - call `model.to(device)` before
    loading if you want the restored model on a specific device, not
    after (see docs/19-gpu-support.md's note on `Adam`/`SGD` snapshotting
    device at construction time - the same ordering caveat applies here).

    Returns:
        The `**metadata` dict passed to `save_checkpoint` (e.g. epoch,
        step, best_val_loss) - the caller's training loop decides what
        to do with it (e.g. resume the step counter).

    Only load checkpoints you trust: this uses `pickle`, which can
    execute arbitrary code when loading a maliciously crafted file - the
    same caveat PyTorch's own `torch.save`/`torch.load` carry.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload["optimizer_state"] is not None:
        optimizer.load_state_dict(payload["optimizer_state"])

    return payload["metadata"]
