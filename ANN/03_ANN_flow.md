## Simple ANN architecture

## Table of Contents

- [Module Class without Explicit backward() in Model](<#module_class_without_explicit_backward()_in_model>)
- [Two Ways to Define a Model](#two_ways_to_define_a_model)
- [Sequential](#sequential)

This code we wrote in the previous section

```python
class MyModel(Module):
    def __init__(self):
        super().__init__()
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

    def backward(self, grad_output):
        grad_output = self.fc3.backward(grad_output)
        grad_output = self.relu2.backward(grad_output)
        grad_output = self.fc2.backward(grad_output)
        grad_output = self.relu1.backward(grad_output)
        grad_output = self.fc1.backward(grad_output)
        return grad_output
```

In PyTorch, this can be written **much more simply** because:

- PyTorch automatically handles backpropagation, so you never need to write a `backward()` method for your model.

- Layers are modules that store their own gradients, and gradients flow automatically through the computation graph.

- You can define models using `nn.Sequential` for compact architectures.

In this section, we will discuss how to write code so that a `backward()` method is **not required** and how to create a `Sequential`

## Module Class without Explicit backward() in Model

In our framework, each layer (like Linear or ReLU) already defines its own backward() method. This means we don’t need to explicitly write a backward() method for the overall model (MyModel) if we organize the layers properly.

```python
class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = xp.zeros_like(data)

class Module:
    def __init__(self):
        self.params = {}
        self.layer_dict = {}     # Add

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.params[name] = value

        if isinstance(value, Module):
            self.layer_dict[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        params = list(self.params.values())
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                params.extend(attr.parameters())
        print('params', params)
        return params

    # Add
    def backward(self, grad_output):
        for layer in reversed(list(self.layer_dict.values())):
            grad_output = layer.backward(grad_output)
        return grad_output
```

```python
# model.__dict__
{'params': {},
 'layer_dict': {'fc1': <__main__.Linear at 0x17208eec520>,
  'relu1': <__main__.ReLU at 0x17208f8c7f0>,
  'fc2': <__main__.Linear at 0x17208f8f0a0>,
  'relu2': <__main__.ReLU at 0x17208eec5e0>,
  'fc3': <__main__.Linear at 0x17208eecbb0>},
 'fc1': <__main__.Linear at 0x17208eec520>,
 'relu1': <__main__.ReLU at 0x17208f8c7f0>,
 'fc2': <__main__.Linear at 0x17208f8f0a0>,
 'relu2': <__main__.ReLU at 0x17208eec5e0>,
 'fc3': <__main__.Linear at 0x17208eecbb0>}
```

The `params` dictionary in the top-level model is empty because submodules are stored in `layer_dict` through the `Module check` inside `__setattr__`:

```python
if isinstance(value, Module):
            self.layer_dict[name] = value
```

When we assign a submodule such as `fc1` (a Linear layer), its own attributes `W` and `b` are `Parameter objects`.
Assigning these attributes triggers the Parameter condition in `__setattr__`, **not** the Module condition:

```python
if isinstance(value, Module):
            self.layer_dict[name] = value
```

```python
# model.__dict__['layer_dict']['fc1'].__dict__
{'params': {'W': <__main__.Parameter at 0x17208f3ebf0>,
  'b': <__main__.Parameter at 0x172085a65c0>},
 'layer_dict': {},
 'deferred_init': False,
 'W': <__main__.Parameter at 0x17208f3ebf0>,
 'b': <__main__.Parameter at 0x172085a65c0>,
 'x': array([[0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        ...,
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.]])}
```

Now do not need a `backward()` method

```python
class MyModel(Module):
    def __init__(self):
        super().__init__()
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

model = MyModel()
```

### How the `backward()` Works in this Architecture

**Module-level backward()** \
The `Module` class provides a generic `backward()` method that loops over all submodules in reverse order `(reversed(list(self.layer_dict.values())))`. \
This acts as a backbone for backpropagation, ensuring gradients flow from the output back to the input through all layers.

**Layer-level backward()** \
Each individual layer, such as `Linear` or `ReLU`, implements its own `backward()` method. \
When `Module.backward()` calls `layer.backward()`, the layer’s own `backward()` is executed. This is what actually computes the gradients for that layer’s parameters.

`MyModel` only defines the forward pass using layers.
It **inherits** `backward()` from `Module`, which automatically propagates gradients through all layers.

This design allows you to **avoid writing manual backward code**, since each layer knows how to backpropagate on its own.

## Sequential

`Sequential` is a container module that allows you to chain layers together **in order**. You can define a neural network by simply passing a list of layers. **The forward pass** automatically applies each layer **in sequence**, and **the backward pass** propagates gradients through all layers in **reverse order**. This avoids the need to manually define the forward and backward methods in your model class.

```python
class Sequential(Module):
    def __init__(self, layers):
        super().__init__()
        self.layer_dict = {}
        for i, layer in enumerate(layers):
            name = "layer" + str(i).zfill(5)
            self.layer_dict[name] = layer


    def forward(self, x):
        # Apply each layer in order
        for i in sorted(self.layer_dict.keys()):
            x = self.layer_dict[i](x)
        return x

    def backward(self, grad_output):
        # Backpropagate through each layer in reverse order
        for i in reversed(sorted(self.layer_dict.keys())):
            grad_output = self.layer_dict[i].backward(grad_output)
        return grad_output

    def parameters(self):
        # Collect parameters from all layers
        params = []
        for layer in self.layer_dict.values():
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params
```

Using

```python
self.layer_dict['layer' + str(i)] = layer
```

can cause ordering issues because string sorting places `layer1, layer10, layer2, …` in the wrong order.
To fix this, we add zero-padding:

```python
name = "layer" + str(i).zfill(5)
```

which produces names like `layer00001`
Zero-padding ensures the layers sort correctly and maintain the intended order.

```python
model_seq = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(32),
    ReLU(),
    Linear(10),
])
```

```python
# model.__dict__
{'params': {},
 'layer_dict': {'layer0': <__main__.Linear at 0x17208f9cb20>,
  'layer1': <__main__.ReLU at 0x17208f9d120>,
  'layer2': <__main__.Linear at 0x17208f9fd00>,
  'layer3': <__main__.ReLU at 0x17208f9fbb0>,
  'layer4': <__main__.Linear at 0x17208f9c7f0>}}
```

```python
# model.__dict__['layer_dict']['layer0'].__dict__
{'params': {'W': <__main__.Parameter at 0x17207f0c220>,
  'b': <__main__.Parameter at 0x17208f9e3b0>},
 'layer_dict': {},
 'deferred_init': False,
 'W': <__main__.Parameter at 0x17207f0c220>,
 'b': <__main__.Parameter at 0x17208f9e3b0>,
 'x': array([[0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        ...,
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.]])}
```

The output is similar to a `Module class without an explicit backward() method`. The only difference is the `layer names` in the keys of layer_dict, but overall it is fine.

### How It Works

**Initialization (`__init__` method)**

- `self.layer_dict` stores the layer instances in order

```python
'layer_dict': {'layer0': <__main__.Linear at 0x17208f9cb20> ...etc.})
```

**Forward Pass (`forward` method)**

- Input `x` is passed through each layer sequentially.
- The output of one layer becomes the input to the next.
- Returns the final output after the last layer.

**Backward Pass (`backward` method)**

- Gradients are propagated in reverse order through the layers.
- Each layer’s own `backward()` handles its parameter gradients automatically.

**Parameters (`parameters` method)**

- Collects all learnable parameters `hasattr(layer, 'parameters')` from each layer `self.layer_dict.values()` then return list of it.

```python
# params
[<__main__.Parameter object at 0x0000017208F8F970>, <__main__.Parameter object at 0x0000017208F446D0>]
```

## Two Ways to Define a Model

In this section, we use a real dataset from `fetch_openml("mnist_784")`.

```python
def dataset(loader_fn, train_num, test_num):
    data_x, data_y = loader_fn

    classes = np.unique(data_y)

    train_x_list = []
    train_y_list = []
    test_x_list = []
    test_y_list = []

    for cls in classes:
        cls_indices = np.where(data_y == cls)[0]
        cls_indices = np.random.permutation(cls_indices)

        X_cls = data_x[cls_indices]
        Y_cls = data_y[cls_indices]

        train_x_list.append(X_cls[:train_num])
        train_y_list.append(Y_cls[:train_num])

        test_x_list.append(X_cls[train_num:train_num + test_num])
        test_y_list.append(Y_cls[train_num:train_num + test_num])

    X_train = np.concatenate(train_x_list)
    y_train = np.concatenate(train_y_list)
    X_test = np.concatenate(test_x_list)
    y_test = np.concatenate(test_y_list)

    train_perm = np.random.permutation(len(X_train))
    X_train = X_train[train_perm]
    y_train = y_train[train_perm]

    test_perm = np.random.permutation(len(X_test))
    X_test = X_test[test_perm]
    y_test = y_test[test_perm]

    return X_train, y_train, X_test, y_test

data = fetch_openml("mnist_784")

# Load and preprocess MNIST dataset
# Normalize pixel values to range 0-1 by dividing by 255.0 (grayscale images originally 0-255)
# Convert labels to integer type using astype('int16') to ensure numeric operations work correctly

X_train, y_train, X_test, y_test = dataset(
    (
        np.asarray(data["data"].values) / 255.0,                 # Normalize input images
        np.asarray(data["target"].values.astype('int16'))        # Convert labels to integers
    ),
    train_num=3000,
    test_num=100
)

```

<img src="utils/mnist_3000_100.png" style="display:block; margin:auto;"/>

This code loads 3,000 samples per class for training and 100 samples per class for testing.

Now, we can define the model in **two ways**:

Using `Sequential`

```python
model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(32),
    ReLU(),
    Linear(10),
])
```

Using a `Custom Class`

```python
class MyModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.relu1 = ReLU()
        self.fc2 = Linear(32)
        self.relu2 = ReLU()
        self.fc3 = Linear(10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model_custom = MyModel()
```

The training code follows the same concept as in the previous section, regardless of which model you use.

```python
# loss_fn = MSE()
loss_fn = CrossEntropy()

epochs = 5
batch_size = 64
initial_lr = 0.01

# Select the model
model = model_seq
# model = model_custom

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
