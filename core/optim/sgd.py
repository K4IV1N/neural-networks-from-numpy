from core.nn.module import Module


class SGD:
    def __init__(self, _module, lr=0.01):
        if isinstance(_module, Module): 
            self._module = _module.parameters()
        else:
            self._module = _module
        self.lr = lr 

    def step(self):
        for param in self._module:
            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self._module:
            param.grad[...] = 0