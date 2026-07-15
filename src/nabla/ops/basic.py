from nabla.function import Function

class Add(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.data + y.data

    def backward(self, grad_output):
        # gradient of addition is 1 for both inputs
        x, y = self.saved_tensors
        grad_x = grad_output
        grad_y = grad_output

        # unbroadcast
        while grad_x.ndim > x.data.ndim:
            grad_x = grad_x.sum(axis=0)
        while grad_y.ndim > y.data.ndim:
            grad_y = grad_y.sum(axis=0)
        
        return grad_x, grad_y
    

class Subtract(Function):
    def forward(self, x, y):
        return x.data - y.data

    def backward(self, grad_output):
        # gradient of subtraction is 1 for the first input and -1 for the second
        return grad_output, -grad_output
    

class Multiply(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.data * y.data

    def backward(self, grad_output):
        x, y = self.saved_tensors
        # gradient of multiplication is y for the first input and x for the second
        return grad_output * y.data, grad_output * x.data
    

class Divide(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.data / y.data

    def backward(self, grad_output):
        x, y = self.saved_tensors
        # gradient of division is 1/y for the first input and -x/y^2 for the second
        return grad_output / y.data, -grad_output * x.data / (y.data ** 2)