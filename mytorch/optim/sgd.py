import numpy as np


class SGD:
    def __init__(self, model, lr=0.01, *, momentum=0.0):
        self.model = model
        self.lr = lr
        self.momentum = momentum 

        if momentum > 0.0:
            velocity = []
            for _, grad in model.params():
                velocity.append(np.zeros_like(grad, dtype=np.float32))
            self.velocity = velocity

    def step(self):
        i = 0
        for param, grad in self.model.params():
            if self.momentum > 0:
                self.velocity[i] *= self.momentum
                self.velocity[i] += grad
                param -= self.velocity[i]
            else:
                param -= self.lr * grad
            i += 1