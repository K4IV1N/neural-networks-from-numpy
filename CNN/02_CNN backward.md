# CNN Backpropagation

## Table of Contents

Now that we have implemented the **forward pass** of the CNN, let's move to **backpropagation**.

# CNN Backpropagation

If you want to know more information how this formula from. i recommned you read this [CNN Backpropagation ](./02_backprop_convolution.md)

$ \frac{\partial L}{\partial b} = sum( \frac{\partial L}{\partial O}) $

$ { \frac{\partial L}{\partial K} = X ⋆ \frac{\partial L}{\partial O} } $

$ \frac{\partial L}{\partial X} = \text{pad}\left(\frac{\partial L}{\partial O}\right) ⋆ \text{rot180}(K) $

```python
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.stride = stride
        self.padding = padding
        self.KH, self.KW = kernel_size

        # He initialization
        scale = np.sqrt(2 / (in_channels * self.KH * self.KW))
        self.W = Parameter(scale * np.random.randn(out_channels, in_channels, self.KH, self.KW))
        self.b = Parameter(np.zeros(out_channels))

        self.x = None

    def forward(self, x):
        self.x = x
        N, C, H, W_in = x.shape
        KH, KW = self.KH, self.KW
        stride, pad = self.stride, self.padding
        F_out = self.W.data.shape[0]

        # Output dimensions
        H_out = (H + 2*pad - KH) // stride + 1
        W_out = (W_in + 2*pad - KW) // stride + 1

        out = np.zeros((N, F_out, H_out, W_out))

        # Pad input
        x_padded = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode="constant")

        # Convolution loops: batch → filters → input channels → height → width
        for n in range(N):
            for f in range(F_out):
                for c in range(C):
                    for i in range(H_out):
                        for j in range(W_out):
                            start_i = i * stride
                            start_j = j * stride
                            region = x_padded[n, c, start_i:start_i+KH, start_j:start_j+KW]
                            out[n, f, i, j] += np.sum(region * self.W.data[f, c])
                out[n, f] += self.b.data[f]

        return out
    def backward(self, d_out):
            B, OC, OH, OW = d_out.shape
            d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, OC)

            self.weights.grad += (d_out_flat.T @ self.cols).reshape(self.weights.data.shape)
            self.biases.grad += d_out_flat.sum(axis=0)
            W_rot = xp.flip(self.weights.data, axis=(2, 3))    # (OC, IC, KH, KW)

            W_rot_col = W_rot.reshape(OC, -1)


            d_cols = d_out_flat @ W_rot_col    # (B*OH*OW, IC*KH*KW)

            dx_padded = col2im(
                d_cols,
                self.x_padded.shape,
                self.kernel_size,
                self.stride,
                (0, 0),
                self.OH,
                self.OW
            )

            # Remove padding before returning
            PH, PW = self.padding
            if PH > 0 or PW > 0:
                return dx_padded[:, :, PH:-PH, PW:-PW]
            return dx_padded
```

## Max Pooling

For each output location $(i,j)$, max pooling selects the maximum value from
its corresponding window $W_{i,j}$

$$
Y_{i,j} = \max_{(m,n) \in W_{i,j}} X_{m,n}
$$

Each window $W_{i,j}$ is defined by the pooling size and stride.

### Backward Pass (Gradient Computation)

For each element $X_{m,n}$ inside the pooling window $W_{i,j}$

$$
\frac{\partial Y_{i,j}}{\partial X_{m,n}} =
\begin{cases}
1, & X_{m,n} = \max\limits_{(u,v)\in W_{i,j}} X_{u,v} \\
0, & \text{otherwise}
\end{cases}
$$

This is implemented as a **mask** indicating the max location:

$$
\text{mask}_{m,n}^{(i,j)} = \mathbf{1}\left(X_{m,n} =
\max_{(u,v)\in W_{i,j}} X_{u,v}\right)
$$

From

$$
\frac{\partial L}{\partial X_{m,n}}
=
\frac{\partial L}{\partial Y_{i,j}}
\cdot
\frac{\partial Y_{i,j}}{\partial X_{m,n}}
$$

Substituting the mask:

$$
\frac{\partial L}{\partial X_{m,n}}
=
\frac{\partial L}{\partial Y_{i,j}}
\cdot
\text{mask}_{m,n}^{(i,j)}
$$

```python
class MaxPool2d(Module):
    def __init__(self, pool_size=(2,2), stride=(1,1)):
        super().__init__()

        # Ensure pool_size and stride are tuples (height, width)
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.pool_size_y, self.pool_size_x = pool_size  # pooling window height and width
        self.stride_y, self.stride_x = stride          # stride in vertical and horizontal directions
        self.x = None   # cache input for backward pass

    def forward(self, x):
        self.x = x      # input array of shape (N, C, H, W)
        N, C, H, W = x.shape

        # Calculate output height and width
        H_out = (H - self.pool_size_y) // self.stride_y + 1
        W_out = (W - self.pool_size_x) // self.stride_x + 1

        # Initialize output array
        out = np.zeros((N, C, H_out, W_out))

        # Loop over batch, channels, and output spatial dimensions
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        # Determine window start and end indices
                        start_i = i * self.stride_y
                        start_j = j * self.stride_x

                        # Extract region of input corresponding to the pooling window
                        region = x[n, c, start_i:start_i+self.pool_size_y,
                                   start_j:start_j+self.pool_size_x]

                        # Store the max value in the output
                        out[n, c, i, j] = np.max(region)

        return out # (N, C, H_out, W_out)

    def backward(self, grad_output):
        x = self.x      # (N, C, H_out, W_out)
        N, C, H, W = x.shape
        H_out, W_out = grad_output.shape[2], grad_output.shape[3]

        # Initialize gradient w.r.t input
        dx = np.zeros_like(x)

        # Loop over batch, channels, and output spatial dimensions
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        # Determine window start and end indices
                        start_i = i * self.stride_y
                        start_j = j * self.stride_x

                        # Extract the same region used in forward pass
                        region = x[n, c, start_i:start_i+self.pool_size_y,
                                   start_j:start_j+self.pool_size_x]

                        # Create mask of max locations
                        max_val = np.max(region)
                        mask = (region == max_val)

                        # Distribute gradient to max locations
                        dx[n, c, start_i:start_i+self.pool_size_y,
                           start_j:start_j+self.pool_size_x] += mask * grad_output[n, c, i, j]

        return dx       # (N, C, H, W)
```

## Flatten

Flatten changes the shape of a tensor from any number of dimensions to 1D. This is useful when the output of a `Conv2d layer` is `4D (N, C, H, W)` and needs to be fed into a `linear layer`, which requires a `1D` input. Therefore, we need to **reshape** the data before passing it to the linear layer.
During backpropagation, Flatten does not change the values, only their layout. Gradients are reshaped back to match the original input tensor.

```python
class Flatten(Module):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        # Flatten all dimensions except batch
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        # Restore original shape
        return grad_output.reshape(self.input_shape)
```

## Combined to train

```python
model = Sequential([
    Conv2d(in_channels=1, out_channels=4, kernel_size=3),
    ReLU(),
    MaxPool2d(pool_size=(2,2), stride=(2,2)),
    Flatten(),
    Linear(32),  # or Linear(4*13*13, 32)
    ReLU(),
    Linear(32, 10)
])
```

the training code still same

```python
loss_fn = MSE()

epochs = 15
batch_size = 32
initial_lr = 0.01

optimizer = SGD(model.parameters(), lr=initial_lr)

for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        x_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        logits = model.forward(x_batch)
        loss = loss_fn.forward(logits, y_batch)
        grad_output = loss_fn.backward()
        model.backward(grad_output)
        optimizer.step()
        optimizer.zero_grad()

    logits_train = model.forward(X_train)
    train_loss = loss_fn.forward(logits_train, y_train)
    train_acc = accuracy(logits_train, y_train)

    logits_test = model.forward(X_test)
    test_loss = loss_fn.forward(logits_test, y_test)
    test_acc = accuracy(logits_test, y_test)

    print(f"Epoch {epoch+1} Summary: "
          f"Train Acc={train_acc:.4f}, Train Loss={train_loss:.4f}, "
          f"Test Acc={test_acc:.4f}, Test Loss={test_loss:.4f}")

```
