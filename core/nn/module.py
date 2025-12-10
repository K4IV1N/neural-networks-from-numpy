import numpy as np
from .parameter import Parameter

class Module:
    def __init__(self):
        self.params = {}
        self.layer_dict = {}  

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.params[name] = value
        super().__setattr__(name, value)

        if isinstance(value, Module):
            self.layer_dict[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        params = list(self.params.values())
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                params.extend(attr.parameters())
        return params

    # Add
    def backward(self, grad_output):
        for layer in reversed(list(self.layer_dict.values())):
            grad_output = layer.backward(grad_output)
        return grad_output