# Artificial Neural Networks (ANN)

## Table of Contents

- [Overview](#overview)
- [Essential Class](#parameter-class)
- [Linear Layer](#linear-layer)
- [Activation Function – ReLU](#activation-function)
- [Loss Function](#loss-function)
- [Simple Forward NN](#simple-forward-nn)

## Overview

This lesson focuses on building the core of a **simple** Artificial Neural Network (ANN)

<img src="utils/ANN%20architecture1.png" alt="ANN Architecture" width="450" style="display:block; margin:auto;"/>

### Model Components

- **Input Layer** – receives data features
- **Hidden Layer** – performs a **linear transformation** followed by an **activation function**
- **Output Layer** – generates final predictions
- **Loss Function** – compares predictions with ground truth labels

---

- Every trainable layer need to store both its **forward values and gradients** (use **`Parameter class`**)
- Layer chaining where the output of one layer becomes the input of the next (use **`Module class`**)

- **Trainable layers** (e.g., Linear): use both `Parameter` and `Modul`e.
- **Activation functions** (e.g., ReLU): inherit only from `Module` since they have no learnable parameters.
- **Loss function**: doesn’t require either Parameter or Module.

---

### Parameter Class

Each trainable layer stores its forward pass value and gradients inside a `Parameter` class:

```python
class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)
```

data → stores the weight or bias during **forward pass** \
grad → stores the gradient computed during **backward pass**

---

### Module class

The layer backbone is designed to allow layers to handle **forward propagation**

```python
class Module:
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)
```

- `__call__` allows you to **call the layer like a function** and automatically triggers `forward` to **chain outputs** to the next layer seamlessly.
- **forward** method, acts as a placeholder that enforces child classes to implement their own forward computation

## Linear Layer

$X_{input}$ has shape **(batch_size, input_features)**\
${W}$ has shape **(output_features, input_features)**\
$b$ has shape **(output_features,)**

```math
C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
```

To perform valid matrix multiplication, the inner dimensions must match.\
For example, a (3, 2) × (2, 4) multiplication works because the inner dimension 2 is the same.

The linear layer computation can be written as either
**${X W^T + b}$** or **${W X^T + b}$** \
PyTorch uses ${X W^T + b}$ so we follow the same convention.

Under the PyTorch convention, users must specify layer parameters as `Linear(in_features, out_features)`, for example `Linear(128, 32)`. This can be inconvenient when the input size is not known beforehand (e.g., after flattening CNN outputs).
To address this, this implementation also supports a lazy mode in which the input size is inferred automatically, for example `Linear(32)`.

```python
class Linear(Module):
    def __init__(self, *args):
        super().__init__()
        # Normal mode: user specifies input and output size
        if len(args) == 2:  # example: Linear(128, 32)
            in_features, out_features = args
            self.deferred_init = False
            self.initialize_params(in_features, out_features)

        # Deferred initialization: Linear(32)
        elif len(args) == 1:
            (out_features,) = args
            self.deferred_init = True
            self.out_features = out_features
            self.W = None
            self.b = None
        else:
            raise ValueError("Linear expects 1 or 2 arguments")


    def initialize_params(self, in_features, out_features):
        # simple
        self.W = Parameter(np.random.randn(in_features, out_features) * 0.01)   # (in_features, out_features, )

        # Kaiming He normal initialization (best for ReLU networks)
        # std = np.sqrt(2.0 / in_features)
        # self.W = Parameter(np.random.randn(in_features, out_features) * std)  # (in_features, out_features, )

        self.b = Parameter(np.zeros(out_features)) # (output_features,)


    def forward(self, x):
        # Deferred initialization
        if self.deferred_init and self.W is None:
            in_features = x.shape[-1]
            self.initialize_params(in_features, self.out_features)
            self.deferred_init = False

        self.x = x
        # x: (batch, in_features)
        return x @ self.W.data.T + self.b.data  # (batch, out_features)

```

## Activation Function

<img src="utils/activation functions.png" alt="ANN Architecture" width="400" style="display:block; margin:auto;"/>

Activation functions introduce **non-linearity** to the network, which allows it to learn complex patterns.  
**`Without`** non-linear activation, stacking multiple layers **collapses** into a **single linear transformation**, so the network **cannot** capture more than linear relationships.

Here, we use the ReLU function:

```python
class ReLU(Module):
    def forward(self, x):
        # Mask for positive elements (positive -> keep, else -> 0)
        self.mask = x > 0
        return x * self.mask
```

## Loss Function

The loss function measures **how well the network's predictions match the true values**.  
It provides a **`single scalar`** value that guides the network during training: \
the smaller the loss, the closer the predictions are to the true targets.

The Mean Squared Error (MSE) specifically computes the **average squared difference** between predicted and true values:

### Mean Squared Error (MSE)

MSE calculates the error by taking the **difference between predictions and true values**, squaring it (so all errors are positive), and then computing the **mean** over all examples.

```math
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2
```

```python
class MSE:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred  # Store predictions for backward pass

        # Convert 1D class labels to one-hot if needed
        if y_true.ndim == 1:
            num_classes = y_pred.shape[1]
            self.y_true = np.eye(num_classes)[y_true]
        else:
            self.y_true = y_true  # Already in proper shape

        # Match dtype with predictions
        self.y_true = self.y_true.astype(y_pred.dtype)

        # Average of squared differences
        loss = np.mean((y_pred - self.y_true) ** 2)
        return loss
```

Although `MSE` is typically used for `regression tasks`, it can be applied to `classification task` if the outputs are **one-hot encoded**.
However, this is **not ideal** because `MSE` treats all differences equally and does not exploit the probabilistic interpretation of class predictions. Gradients can be smaller and learning slower.

### Cross-Entropy Loss

Cross-Entropy Loss is designed specifically for `classification tasks`. It compares the true class distribution with the predicted probability distribution. \
For a single example, let the predicted probabilities be $ \hat{y}\_k $ and the true labels be $ y_k $. The Cross-Entropy Loss is defined as:

```math
\text{CrossEntropy Loss} = - \sum_{k=1}^{K} y_k \, \log(\hat{y}_k)
```

For a batch of $N$ examples, we take the **average over all examples**:

```math
L = - \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{y}_k^{(i)})
```

This formula is used in training `classification models`, like MNIST digit classifiers, to compute the loss over a batch of images.

Notes

- Typically, the network outputs are passed through a **softmax** function to convert them into probabilities.
- Cross-Entropy Loss produces larger, more informative gradients when predictions are far from the correct class. This often results in **faster and more stable training** compared to other loss functions.

```python
class CrossEntropy:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred # Store predictions for backward pass

        # If labels are 1D (class indices), convert them to one-hot encoding
        if y_true.ndim == 1:
            num_classes = y_pred.shape[1]  # Number of output classes
            self.y_true = np.eye(num_classes)[y_true]  # One-hot encode
        else:
            self.y_true = y_true  # Already one-hot encoded

        # Match dtype with predictions
        self.y_true = self.y_true.astype(y_pred.dtype)

        # Clip predictions to avoid log(0) which can cause numerical issues
        y_pred_clipped = np.clip(y_pred, 1e-12, 1.0)

        # Compute cross-entropy loss:
        #   - sum over classes for each sample
        #   - then average over all samples
        loss = -np.mean(np.sum(self.y_true * np.log(y_pred_clipped), axis=1))
        return loss
```

Sometimes, **y_true** is provided as a `1D array` of class indices, e.g., `y_true = [0, 2, 1, 2]`.  
To compute the MSE correctly with **y_pred**, **y_true** must have the same shape as **y_pred**, which means converting it to a **2D one-hot encoded array**:

```math
[0, 2, 1, 2] →
\begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
```

## Simple Forward NN

Combine all components and test the forward pass with the loss function.

```python
class MyModel:
    def __init__(self):
        self.fc1 = Linear(784, 128)
        self.relu1 = ReLU()
        self.fc2 = Linear(128, 32)
        self.relu2 = ReLU()
        self.fc3 = Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

batch_size = 10     # Number of samples
num_classes = 10    # Number of class

loss_fn = MSE()
# loss_fn = CrossEntropy()


# input (random images)
X_random = np.random.rand(batch_size, 784)   # shape: (10,784)

# labels (random class indices 0–9)
y_indices = np.random.randint(0, num_classes, size=batch_size)

# y_batch = np.eye(num_classes)[y_indices]
# Not needed because we defined the loss to handle both label encoding and one-hot encoding.

model = MyModel()   # Instantiate the model

# Run forward pass
logits = model.forward(X_random)
loss = loss_fn.forward(logits, y_indices)

print("Output shape:", logits.shape)
print("Random labels:", y_indices)
print("logits:", logits[0])
print("Predicted labels:", np.argmax(logits, axis=1)) # Logits give class scores -> use np.argmax to convert to labels
print("Loss:", loss)
```

Result:

```
Output shape: (10, 10)
Random labels: [6 5 8 6 5 3 7 9 1 7]
logits: [ 5.38458944e-04 -1.85460259e-04  9.33223835e-04  2.91333401e-04
 -1.24151116e-04  2.26516581e-04 -3.89904346e-04  6.81347679e-05
 -3.21200608e-04  2.60717170e-04]
Predicted labels: [2 0 0 0 0 0 0 2 0 2]
Loss: 0.10001136097849546
```
