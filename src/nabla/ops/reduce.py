from nabla.function import Function
import numpy as np

class Sum(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.sum(x.data)
    
    def backward(self, grad_output):
        x, = self.saved_tensors
        # gradient of sum is 1 for each element in the input tensor
        # so basically, we just blow up the grad_output to the shape of x.data
        return (np.ones_like(x.data) * grad_output,)
    

class Mean(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.mean(x.data)
    
    def backward(self, grad_output):
        x, = self.saved_tensors
        # gradient of mean is 1/n for each element in the input tensor
        n = x.data.size
        return (np.ones_like(x.data) * (grad_output / n),)