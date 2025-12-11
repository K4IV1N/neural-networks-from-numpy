from .module import Module
import numpy as np

class MaxPool2d(Module):
    def __init__(self, pool_size=(2,2), stride=None):
        super().__init__()
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        self.pool_size = pool_size

        if stride is None:
            stride = pool_size 
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        self.x = None
        self.argmax = None

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        KH, KW = self.pool_size
        SH, SW = self.stride

        OH = (H - KH) // SH + 1
        OW = (W - KW) // SW + 1

        shape = (N, C, OH, OW, KH, KW)
        strides = (
            x.strides[0],
            x.strides[1],
            SH * x.strides[2],
            SW * x.strides[3],
            x.strides[2],
            x.strides[3],
        )

        windows = np.lib.stride_tricks.as_strided(
            x, shape=shape, strides=strides, writeable=False
        ) 
        windows_reshaped = windows.reshape(N, C, OH, OW, KH * KW)
        
        out = windows_reshaped.max(axis=4)
        self.argmax = windows_reshaped.argmax(axis=4)
        return out 

    def backward(self, grad_output):
        x = self.x
        N, C, H, W = x.shape
        KH, KW = self.pool_size
        SH, SW = self.stride
        OH, OW = grad_output.shape[2:]

        dx = np.zeros_like(x)

        for i in range(OH):
            for j in range(OW):
                idx = self.argmax[:, :, i, j]
                kh = idx // KW
                kw = idx % KW
                ih = i * SH + kh
                iw = j * SW + kw

                dx[np.arange(N)[:, None], np.arange(C), ih, iw] += grad_output[:, :, i, j]
        return dx