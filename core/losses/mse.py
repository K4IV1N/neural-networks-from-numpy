import numpy as np

class MSE:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred 

        if y_true.ndim == 1:
            num_classes = y_pred.shape[1]
            self.y_true = np.eye(num_classes)[y_true]
        else:
            self.y_true = y_true  

        self.y_true = self.y_true.astype(y_pred.dtype)

        loss = np.mean((y_pred - self.y_true) ** 2)
        return loss
    
    def backward(self): 
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]