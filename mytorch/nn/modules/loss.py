import numpy as np


class MSELoss:
    def forward(self, A, Y):
        self.A = A
        self.Y = Y
        return ((A - Y) * (A - Y)).mean()
    
    def backward(self):
        return (2 / np.prod(self.A.shape)) * (self.A - self.Y)
    

class CrossEntropyLoss:
    def forward(self, A, Y):
        self.Y = Y
        expA = np.exp(A)
        self.softA = expA / expA.sum(axis=1, keepdims=True)
        return - (Y * np.log(self.softA)).mean()
    
    def backward(self):
        return self.softA - self.Y