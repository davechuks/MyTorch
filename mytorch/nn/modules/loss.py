import numpy as np


class MSELoss:
    """Implements the mean squared error loss."""

    def forward(self, A, Y):
        """
        Computes and returns the mean squared error between an estimated value
          and the desired value.

        Args:
            A (numpy array): the estimated value.
            Y (numpy array): the desired value. 
        """

        self.A = A
        self.Y = Y
        return ((A - Y) * (A - Y)).mean()

    def backward(self):
        """
        Computes and returns the derivative of the mean squared error with respect to the estimated value.
        """

        return (2 / np.prod(self.A.shape)) * (self.A - self.Y)


class CrossEntropyLoss:
    """Implements the cross-entropy loss."""

    def forward(self, A, Y):
        """
        Computes and returns the cross-entropy loss between two discrete probability distributions.

        Args:
            A (numpy array): the raw output logits.
            Y (numpy array): a one-hot encoded desired probability. 
        """

        self.Y = Y
        expA = np.exp(A)
        self.softA = expA / expA.sum(axis=1, keepdims=True)
        return -(Y * np.log(self.softA)).sum() / Y.shape[0]

    def backward(self):
        """
        Computes and returns the derivative of the loss with respect to the input probability.
        """
        return (self.softA - self.Y) / self.Y.shape[0]