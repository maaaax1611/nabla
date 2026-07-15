import numpy as np
from nabla.ops.basic import Add, Subtract, Multiply, Divide
from nabla.ops.reduce import Sum, Mean
from nabla.ops.transform import MatMul
from nabla.ops.activation import ReLU, Sigmoid

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None
        self._prev = []

    def backward(self, grad=None):
        # check if we already have a gradient
        # if not initialize it with ones
        if grad is None:
            self.grad = np.ones_like(self.data)
        else:
            self.grad = grad

        # topo sort graph
        topo = []
        visited = set()
        def topo_sort(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    topo_sort(parent)
                topo.append(tensor)
        topo_sort(self)

        # go backwad through the sorted graph
        for tensor in reversed(topo):
            # skip leaf tensors
            if tensor._ctx:
                # how does the current gradient affect the parents?
                # this should give us a tuple/list of gradients for each parent
                grads = tensor._ctx.backward(tensor.grad)
                for parent, grad in zip(tensor._prev, grads):
                    if parent.requires_grad:
                        if parent.grad is None:
                            parent.grad = grad
                        else:
                            parent.grad += grad

    def __add__(self, other):
        return Add.apply(self, other)
    
    def __sub__(self, other):
        return Subtract.apply(self, other)
    
    def __mul__(self, other):
        return Multiply.apply(self, other)
    
    def __truediv__(self, other):
        return Divide.apply(self, other)
    
    def __matmul__(self, other):
        return MatMul.apply(self, other)
    
    def sum(self):
        return Sum.apply(self)
    
    def mean(self):
        return Mean.apply(self)
    
    def matmul(self, other):
        return MatMul.apply(self, other)
    
    def relu(self):
        return ReLU.apply(self)
    
    def sigmoid(self):
        return Sigmoid.apply(self)