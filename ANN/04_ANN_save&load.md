# Saving and Loading a Trained Model

In this section, we'll discuss how to `save` and `load` a trained model.

### Saving the Model

The key is saving only the **trainable parameters** (weights and biases) of layers such as Linear and Convolutional, Embedding layer, etc.

model parameters can be saved in formats like `.npy` (NumPy), `.pth` (PyTorch), or `.h5` (TensorFlow). For instance:

- In **PyTorch**, save the model's state dictionary.
  This dictionary contains the **layer names** as keys and their corresponding **parameter tensors** (weights, biases, etc.) as values.

```
{
  'fc1.weight': tensor([[...]]),
  'fc1.bias': tensor([...]),
  ...
}
```

```python
torch.save(model.state_dict(), "model.pth")
```

- In **TensorFlow/Keras**, save both the architecture and weights:

  ```python
  model.save('my_model.h5')
  ```

**Note**: These methods save only the model parameters and architecture. **Hyperparameters** (like **optimizer**, **loss function**, and **learning rate scheduler**) are **not saved** and need to be manually specified when loading the model.

### Loading the Model

In **PyTorch**, loading a model is tricky because it only saves the **model weights**, not the architecture. You must **recreate the model architecture** before loading the saved parameters:

```python
model = MyModel()  # Recreate the architecture
model.load_state_dict(torch.load("model.pth"))  # Load saved weights
```

In **TensorFlow/Keras**, both the architecture and weights are saved together, so you can load the model in one step:

```python
loaded_model = tf.keras.models.load_model('my_model.h5')
```

However, **hyperparameters** (optimizer, loss, scheduler) are still **not saved**. After loading the model, you must **recompile** it with the appropriate settings before continuing training or evaluation.

in this section, we will dicuss how to make simple custrom model save in npy format (model parameter) like pytorch

## Save model

from previous section. you can see the model.**dict** give us

```python
# Sequential
{'params': {},
 'layer_dict': {'layer0': <__main__.Linear at 0x17208f9cb20>,
  'layer1': <__main__.ReLU at 0x17208f9d120>,
  'layer2': <__main__.Linear at 0x17208f9fd00>,
  'layer3': <__main__.ReLU at 0x17208f9fbb0>,
  'layer4': <__main__.Linear at 0x17208f9c7f0>}}


# Module Class without Explicit backward() in Model
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

The output is similar to a `Module class without an explicit backward() method`. The only difference is the `layer names` in the keys of layer_dict, but overall it is fine.

you can see, both model architecture store its layer in layer_dict, so we need to access it. due to we only learnable like linear or

```python
def save_model(model, path):
    state = {}
    for module_name, module in model.layer_dict.items():
        if isinstance(module, Module):
            for pname, p in module.params.items():
                state[f"{module_name}.{pname}"] = p.data

    for k, v in state.items():
        print(f"{k}: shape={v.shape}")
    np.save(path, state)

```

```python
# Sequential
layer0.W: shape=(784, 128)
layer0.b: shape=(128,)
layer2.W: shape=(128, 32)
layer2.b: shape=(32,)
layer4.W: shape=(32, 10)
layer4.b: shape=(10,)


# Module Class without Explicit backward() in Model
fc1.W: shape=(784, 128)
fc1.b: shape=(128,)
fc2.W: shape=(128, 32)
fc2.b: shape=(32,)
fc3.W: shape=(32, 10)
fc3.b: shape=(10,)
```

## Load model

```python
def load_model(model, path):
    # Load the saved file
    state = np.load(path, allow_pickle=True).item()

    # Update the parameters of the current model
    for module_name, module in model.layer_dict.items():
        if isinstance(module, Module):
            for pname, p in module.params.items():
                key = f"{module_name}.{pname}"
                if key in state:
                    p.data = state[key]  # Update its value
                else:
                    print("Missing:", key)

    return model
```

## Usage

After you train your `model`:

```python
# save
save_path = 'saved_1.npy'
save_model(model, save_path)

# load
model_load = load_model(model, save_path)
```

You can continue training `model_load` using the same training code as before saving.
