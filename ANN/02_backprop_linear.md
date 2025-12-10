## Linear Backpropagation

For the linear layer: ${Z = X W + b}$

$${Z_{N,D_{out}} = X_{N,D_{in}} (W_{D_{in},D_{out}})^T + b_{out}}$$

- ${N}$: batch size
- ${D_{\text{in}}}$: input feature dimension
- ${D_{\text{out}}}$: input feature dimension

During backpropagation, we compute gradients with respect to three variables `W, b, X`:

- W: for updating the layer’s **weights**
- b: for updating the layer’s **biases**
- X: the input; we return $dL/dX$ as the `output` to propagate the error to the next layer

### Gradient with respect to `Weights`

---

The linear layer is:
$$ Z = X W + b $$

From matrix multiplication:

```math
Z_{n,j} = \sum_i X_{n,i} (W)_{i,j} + b_j
```

Expanded:

```math
Z_{n,j} = X_{n,1} W_{1,j} + X_{n,2} W_{2,j} + \dots + X_{n,i} W_{i,j} + \dots + b_j
```

Only `one term` in this sum contains ${W_{i,j}}$ that is ${X_{n,i} W_{i,j}}$

Example for a specific sample ${n = 0}$ and output index ${j = 3}$:

$$
Z_{0,3} = X_{0,1} W_{1,3} + X_{0,2} W_{2,3} + \dots + X_{0,i} W_{i,3} + \dots + b_3
$$

Only the term ${X_{0,i} W_{i,3}}$ depends on ${W_{i,3}}$ → $\frac{\partial Y_{0,3}}{\partial W_{i,3}}= X_{0,i}$

Therefore:

```math
\frac{\partial Z_{n,j}}{\partial W_{i,j}} = X_{n,i}
```

The gradient of the loss $`L`$ with respect to the weights $`W_{i,j}`$ is:

```math
\frac{\partial L}{\partial W_{i,j}}
= \sum_{n=1}^{N} \frac{\partial L}{\partial Z_{n,j}} \frac{\partial Z_{n,j}}{\partial W_{i,j}}
```

```math
\frac{\partial L}{\partial W_{i,j}} = \sum_{n=1}^{N} \frac{\partial L}{\partial Z_{n,j}} (X_{n,i})
```

Since $\frac{\partial L}{\partial Z_{n,j}}$ has shape $(n, j)$, we take `transpose` to `X` for align the dimensions for proper matrix multiplication.

In **matrix form**, this can be written as:

```math
\frac{\partial L}{\partial W} =  X^TdZ  \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}
```

### Gradient with respect to `Bias`

---

Each bias term $b_j$ contributes equally to all samples, so:

```math
\frac{\partial Z_{n,j}}{\partial b_j} = \frac{\partial}{\partial b_j} \Big( \sum_i X_{n,i} W_{i,j} + b_j \Big) = \frac{\partial}{\partial b_j} \Big(  b_j \Big) = 1
```

Hence, the gradient of the loss with respect to the bias is:

```math
\frac{\partial L}{\partial b_j} = \sum_{n=1}^{N} \frac{\partial L}{\partial Z_{n,j}}\frac{\partial Z_{n,j}}{\partial b_j}  =\sum_{n=1}^{N} \frac{\partial L}{\partial Z_{n,j}}(1)  =\sum_{n=1}^{N} \frac{\partial L}{\partial Z_{n,j}}
```

In **matrix form**, this can be written as:

```math
\frac{\partial L}{\partial b} = \sum_{n=1}^{N} dZ = \text{sum}(dZ, \text{axis}=0) \in \mathbb{R}^{D_{\text{out}}}
```

`axis = 0` meaning sum over the batch dimension

### Gradient with respect to `Input`

---

propagate the gradient backward to the previous layer:

$$
\frac{\partial L}{\partial X_{n,i}}
= \sum_{j}
\frac{\partial L}{\partial Z_{n,j}}
\frac{\partial Z_{n,j}}{\partial X_{n,i}}
$$

from

```math
Z_{n,j} = X_{n,1} W_{1,j} + X_{n,2} W_{2,j} + \dots + X_{n,i} W_{i,j} + \dots + b_j
```

Only `one term` in this sum contains $X_{n,i}$ that is ${X_{n,i} W_{i,j}}$

Therefore:
$$\frac{\partial Z_{n,j}}{\partial X_{n,i}} = W_{i,j}$$

$$
\frac{\partial L}{\partial X_{n,i}}
= \sum_{j} \frac{\partial L}{\partial Z_{n,j}}(W_{i,j})
$$

Since $\frac{\partial L}{\partial Z_{n,j}}$ has shape $(n, j)$, we take `transpose` to $W_{i,j}$ for proper matrix multiplication.  
In matrix form, this can be written as:
$$ \frac{\partial L}{\partial X} = dZ \, W^T $$

### Summary

---

Forward \
${Z = X W + b}$

Backward \
$\frac{\partial L}{\partial W} = X^T dZ $

$\frac{\partial L}{\partial b} = \text{sum}(dZ, \text{axis}=0)$

$\frac{\partial L}{\partial X} = dZ \, W^T$
