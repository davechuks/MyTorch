import numpy as np


class Identity:
    def forward(self, Z):
        self.A = Z
        return Z

    def backward(self):
        return np.ones_like(self.A, dtype=np.float32)
    

class Sigmoid:
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A
    
    def backward(self):
        return self.A(1 - self.A)
    
class Tanh:
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A
    
    def backward(self):
        return 1 - self.A ** 2
    
class ReLU:
    def forward(self, Z):
        self.A = np.maximum(0, Z)
        return self.A
    
    def backward(self):
        return np.where(self.A > 0, 1, 0)
