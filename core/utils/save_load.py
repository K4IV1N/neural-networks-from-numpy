from core.nn.module import Module
import numpy as np

def save_model(model, path):
    state = {}
    for module_name, module in model.layer_dict.items():
        if isinstance(module, Module):
            for pname, p in module.params.items():
                state[f"{module_name}.{pname}"] = p.data

    for k, v in state.items():
        print(f"{k}: shape={v.shape}")
    np.save(path, state)

def load_model(model, path):
    state = np.load(path, allow_pickle=True).item()

    for module_name, module in model.layer_dict.items():
        if isinstance(module, Module):
            for pname, p in module.params.items():
                key = f"{module_name}.{pname}"
                if key in state:
                    p.data = state[key]  
                else:
                    print("Missing:", key)

    return model