import numpy as np

class CrossEntropy:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred 

        if y_true.ndim == 1:
            num_classes = y_pred.shape[1] 
            self.y_true = np.eye(num_classes)[y_true] 
        else:
            self.y_true = y_true  

        self.y_true = self.y_true.astype(y_pred.dtype)
        y_pred_clipped = np.clip(y_pred, 1e-12, 1.0)
        
        loss = -np.mean(np.sum(self.y_true * np.log(y_pred_clipped), axis=1))
        return loss
    
    def backward(self):
        N = self.y_pred.shape[0]  
        grad = (self.y_pred - self.y_true) / N
        return grad