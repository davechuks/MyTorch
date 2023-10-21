import numpy as np
from layer import Layer


class Linear(Layer):
    """A layer that applies an affine transformation to the input."""

    def __init__(self, in_features, out_features, bias=True):
        """
        Creates a layer that applies a linear transformation to the inputs.

        Args:
            in_features (int): the size of each input sample.
            out_features (int): the size of each output sample.
            bias (boolean): should the layer add a bias term to the transformation? Default: True.
        """

        self.W = np.zeros((out_features, in_features), dtype=np.float32)
        self.dLdW = np.zeros((out_features, in_features), dtype=np.float32)
        self.bias = bias
        if bias:
            self.b = np.zeros((1, out_features), dtype=np.float32)
            self.dLdb = np.zeros((1, out_features), dtype=np.float32)

    def forward(self, A):
        """
        Applies a linear transformation to the input.

        Args:
            A (numpy array): the input. The shape should be N * C_in where N is the
                number of samples and C_in is the number of input features. It returns
                an array of shape N * C_out where C_out is the number of output features.
        """
        self.A = A
        return A.dot(self.W.T) + (self.b if self.bias else 0)

    def backward(self, dLdZ):
        self.dLdW = (dLdZ[..., np.newaxis] * self.A[:, np.newaxis, :]).sum(axis=0)
        if self.bias:
            self.dLdb = dLdZ.sum(axis=0, keepdims=True)
        return dLdZ.dot(self.W)
    
    def params(self):
        yield [self.W, self.dLdW]
        if self.bias:
            yield [self.b, self.dLdb]