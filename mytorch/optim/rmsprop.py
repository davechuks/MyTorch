import numpy as np


class RMSProp:
    """Implements the root-mean-square propagation algorithm."""
    
    def __init__(self, model, lr=0.001, *, beta=0.9, epsilon=1e-8):
        self.model = model
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon

        force = []
        for _, grad in model.params():
            force.append(np.zeros_like(grad))
        self.force = force

    def step(self):
        i = 0
        for param, grad in self.model.params():
            self.force[i] *= self.beta
            self.force[i] += (1 - self.beta) * grad**2

            param -= (self.lr / np.sqrt(self.force[i] + self.epsilon)) * grad
            i += 1