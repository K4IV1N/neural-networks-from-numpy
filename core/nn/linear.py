from .module import Module
from .parameter import Parameter
import numpy as np

class Linear(Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 2:  
            in_features, out_features = args
            self.deferred_init = False
            self.initialize_params(in_features, out_features)

        # Deferred initialization
        elif len(args) == 1:
            (out_features,) = args
            self.deferred_init = True
            self.out_features = out_features
            self.W = None
            self.b = None
        else:
            raise ValueError("Linear expects 1 or 2 arguments")


    def initialize_params(self, in_features, out_features):
        # Kaiming He normal initialization
        std = np.sqrt(2.0 / in_features)                    
        self.W = Parameter(np.random.randn(in_features, out_features) * std) 
        
        self.b = Parameter(np.zeros(out_features))


    def forward(self, x):
        # Deferred initialization
        if self.deferred_init and self.W is None:
            in_features = x.shape[-1]
            self.initialize_params(in_features, self.out_features)
            self.deferred_init = False

        self.x = x
        return x @ self.W.data + self.b.data   
    
    def backward(self, grad_output):
        self.W.grad += self.x.T @ grad_output     
        self.b.grad += grad_output.sum(axis=0)
        return grad_output @ self.W.data.T         