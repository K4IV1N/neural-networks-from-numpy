from .module import Module

class ReLU(Module):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_input):
        return grad_input * self.mask