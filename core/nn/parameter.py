import numpy as np

class Parameter:
    def __init__(self, data):
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(data, dtype=np.float32)
