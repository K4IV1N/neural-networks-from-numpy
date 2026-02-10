import numpy as np
from .module import Module

class Softmax(Module):
    def forward(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, grad_input):
        batch_size, _ = grad_input.shape
        grad = np.zeros_like(grad_input)

        for i in range(batch_size):
            y = self.out[i].reshape(-1, 1)        
            jacobian = np.diagflat(y) - y @ y.T     
            grad[i] = jacobian @ grad_input[i]

        return grad
