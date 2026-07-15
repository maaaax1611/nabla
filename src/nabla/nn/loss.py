from nabla.nn.module import Module
from nabla.tensor import Tensor

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        if not isinstance(predictions, Tensor) or not isinstance(targets, Tensor):
            raise TypeError("Both predictions and targets must be instances of Tensor.")
        if predictions.data.shape != targets.data.shape:
            raise ValueError(f"Predictions and targets must have the same shape. Got {predictions.data.shape} and {targets.data.shape}.")
        
        diff = predictions - targets
        return (diff * diff).mean()