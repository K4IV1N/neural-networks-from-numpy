from .module import Module
from .parameter import Parameter
import numpy as np

def im2col(input_data, kernel_size, stride, padding):
    B, C, H, W = input_data.shape
    KH, KW = kernel_size
    SH, SW = stride
    PH, PW = padding

    H_p, W_p = input_data.shape[2], input_data.shape[3]
    OH = (H_p - KH) // SH + 1
    OW = (W_p - KW) // SW + 1

    col = np.empty((B, OH, OW, C, KH, KW), dtype=input_data.dtype)
    for y in range(OH):
        y_min = y * SH
        y_max = y_min + KH
        for x in range(OW):
            x_min = x * SW
            x_max = x_min + KW
            col[:, y, x, :, :, :] = input_data[:, :, y_min:y_max, x_min:x_max]

    return col.reshape(B * OH * OW, -1), OH, OW


def col2im(cols, input_shape, kernel_size, stride, padding, OH, OW):
    B, C, H, W = input_shape
    KH, KW = kernel_size
    SH, SW = stride
    PH, PW = padding

    H_p, W_p = H + 2*PH, W + 2*PW
    cols_reshaped = cols.reshape(B, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
    out = np.zeros((B, C, H_p, W_p), dtype=cols.dtype)
    for y in range(OH):
        y_start = y * SH
        for x in range(OW):
            x_start = x * SW
            out[:, :, y_start:y_start+KH, x_start:x_start+KW] += cols_reshaped[:, :, :, :, y, x]

    if PH > 0 or PW > 0:
        return out[:, :, PH:H_p-PH, PW:W_p-PW]
    return out

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        KH, KW = self.kernel_size

        limit = 1 / np.sqrt(in_channels * KH * KW)
        self.weights = Parameter(np.random.uniform(-limit, limit, (out_channels, in_channels, KH, KW)))
        self.biases = Parameter(np.zeros(out_channels))

    def forward(self, x):
        self.x = x
        PH, PW = self.padding
        if PH > 0 or PW > 0:
            self.x_padded = np.pad(x, ((0, 0), (0, 0), (PH, PH), (PW, PW)), mode='constant')
        else:
            self.x_padded = x

        self.cols, self.OH, self.OW = im2col(
            self.x_padded, self.kernel_size, self.stride, (0, 0)
        )

        W_col = self.weights.data.reshape(self.out_channels, -1).T
        out = self.cols @ W_col
        out += self.biases.data
        out = out.reshape(x.shape[0], self.OH, self.OW, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, d_out):
        B, OC, OH, OW = d_out.shape
        d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, OC)

        self.weights.grad += (d_out_flat.T @ self.cols).reshape(self.weights.data.shape)
        self.biases.grad += d_out_flat.sum(axis=0)
        W_rot = np.flip(self.weights.data, axis=(2, 3))
        W_rot_col = W_rot.reshape(OC, -1)
        d_cols = d_out_flat @ W_rot_col

        dx_padded = col2im(
            d_cols,
            self.x_padded.shape,
            self.kernel_size,
            self.stride,
            (0, 0),
            self.OH,
            self.OW
        )

        PH, PW = self.padding
        if PH > 0 or PW > 0:
            return dx_padded[:, :, PH:-PH, PW:-PW]
        return dx_padded
