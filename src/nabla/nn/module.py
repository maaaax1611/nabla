from nabla.tensor import Tensor

class Module:
    def __init__(self):
        pass

    def parameters(self):
        """
        Returns a list of all trainable parameters in the module and its submodules.
        """
        params = []
        for value in self.__dict__.values():
            # handle raw tensors 
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            # recursively extract parameters from modules and submodules
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params
    
    def zero_grad(self):
        """"
        Sets the gradients of all trainable parameters in the module and its submodules to zero.
        """
        for param in self.parameters():
            param.grad = None   # a parameter is a tensor object

    def __call__(self, *args):
        return self.forward(*args)
    
    def forward(self, *args):
        raise NotImplementedError("Subclasses must implement the forward method.")