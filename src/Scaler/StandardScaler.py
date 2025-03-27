import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0 
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def inverse_transform(self, X):
        return X * self.std + self.mean