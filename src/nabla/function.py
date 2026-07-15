class Function:
    """
    Base class for all differentiable operations.

    All custom operations should inherit from this class
    and implement `forward` and `backward`

    Attributes:
        saved_tensors: tensors saved during forward for use in backward
    """
    def __init__(self):
        self.saved_tensors = []

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def forward(self, *args):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *inputs):
        from nabla.tensor import Tensor
        ctx = cls()             # create a new op-instance e.g. add, mul, etc.
        result = ctx.forward(*inputs)  # compute forward pass of new op-instance
        out = Tensor(result)    # wrap it inside a Tensor

        out._ctx = ctx          # what op created current instance
        out._prev = inputs      # list of input tensors

        if any(t.requires_grad for t in inputs):
            out.requires_grad = True

        return out
