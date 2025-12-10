import numpy as np

def accuracy(logits, targets):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == targets)