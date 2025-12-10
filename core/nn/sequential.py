from .module import Module

class Sequential(Module):
    def __init__(self, layers):
        super().__init__()
        self.layer_dict = {}        
        for i, layer in enumerate(layers):
            self.layer_dict['layer' + str(i)] = layer

    def forward(self, x):
        for i in sorted(self.layer_dict.keys()):
            x = self.layer_dict[i](x)
        return x

    def backward(self, grad_output):
        for i in reversed(sorted(self.layer_dict.keys())):
            grad_output = self.layer_dict[i].backward(grad_output)
        return grad_output

    def parameters(self):
        params = []
        for layer in self.layer_dict.values():
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params