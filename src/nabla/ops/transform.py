from nabla.function import Function
import numpy as np

class MatMul(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return np.dot(a.data, b.data)
    
    def backward(self, grad_output):
        a, b = self.saved_tensors
        # gradient of matrix multiplication is b^T for the first input and a^T for the second
        grad_a = np.dot(grad_output, b.data.T)
        grad_b = np.dot(a.data.T, grad_output)
        return grad_a, grad_b