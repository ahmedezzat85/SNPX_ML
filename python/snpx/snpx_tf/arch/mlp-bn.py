""" mlp --> Multi-Layer Perceptron
"""
from __future__ import absolute_import

import numpy as np
from .net import SNPXNet

def snpx_net_create(num_classes, fp16=False, n_hidden_layers=2, n_hidden_units=256, act_fn='relu'):
    """
    Create a Softmax Multi-Layer Perceptron Classifier in mxnet.

    Parameters
    ----------
    num_classes : int
        Number of output Classes
    n_hidden_layers : int
        Number of Hidden Layers
    n_hidden_units : int
        Number of neuorons in each hidden layer
    act_fn : str
        Type of activation function for the each hidden neuoron

    Return
    ------
    MLP : Symbol
        Network Symbol for the multi-layer perceptron

    """
    snpx_mlp = SNPXNet(fp16=fp16, batch_norm=True)
    snpx_mlp.Flatten()
    for i in range(0, n_hidden_layers):
        snpx_mlp.FullyConnected(num_hidden=n_hidden_units, name="FC"+str(i+1), act=act_fn)

    snpx_mlp.Softmax(num_classes)
    return snpx_mlp.Net