from nabla.function import Function
import numpy as np

class ReLU(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.maximum(0, x.data)

    def backward(self, grad_output):
        x, = self.saved_tensors
        grad_x = grad_output * (x.data > 0)
        return (grad_x,)
    

class Sigmoid(Function):
    def forward(self, x):
        sigmoid = 1 / (1 + np.exp(-x.data))
        self.save_for_backward(sigmoid)
        return sigmoid
    
    def backward(self, grad_output):
        sigmoid, = self.saved_tensors
        grad_x = grad_output * sigmoid * (1 - sigmoid)
        return (grad_x,)
    