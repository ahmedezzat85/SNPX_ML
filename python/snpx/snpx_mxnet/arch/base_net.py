""" Neural Network Operations
"""

import mxnet as mx
import numpy as np

class SNPXBaseNet(object):
    def __init__(self, fp16=False):
        self.fp16   = fp16

    def Convolution(self, name, num_filter, kernel, act='relu', stride=None, pad=(), batch_norm=False):
        """ Perform 2D or 3D Convolution per kernel size.

        Parameters
        ----------
        name : str
            Friendly name of the convolution layer.
        num_filter : int, number
            Number of convolution filters per this layer.
        kernel : tuple
            Convolution filter dimensions (2 or 3 elements for 2D or 3D convolution)
        act : str
            Activation function (non-linearity) applied to the convolution output. 
        """
        raise NotImplementedError()

    def FullyConnected(self, num_hidden, name, act=None, batch_norm=False):
        """
        """
        raise NotImplementedError()

    def Softmax(self, num_classes):
        """
        """
        raise NotImplementedError()

    def Flatten(self):
        raise NotImplementedError()
    
    def MaxPool(self, name, kernel, stride):
        raise NotImplementedError()

    def AvgPool(self, name, kernel, stride):
        raise NotImplementedError()

    def Dropout(self):
        raise NotImplementedError()

    def ZeroPadding(self, pad_size):
        raise NotImplementedError()

    def Concat(self, L=[]):
        raise NotImplementedError()