import numpy as np
from core.nn.module import Module


class Adam:
    def __init__(
        self,
        _module,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8
    ):
        if isinstance(_module, Module):
            self.params = _module.parameters()
        else:
            self.params = _module

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        self.t = 0

        self.m = []
        self.v = []

        for p in self.params:
            self.m.append(np.zeros_like(p.data))
            self.v.append(np.zeros_like(p.data))

    def step(self):
        self.t += 1

        for i, p in enumerate(self.params):
            g = p.grad

            # First moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

            # Second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad[...] = 0
