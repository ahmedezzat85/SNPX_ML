""" cnn12

12 Layers CNN with average pooling feature layer
"""
from __future__ import absolute_import

import numpy as np
from .net import SNPXNet

def snpx_net_create(num_classes, fp16=False):
    """
    """
    snpx_c3d = SNPXNet(fp16=fp16)
    # CONV1 3x3x3/1,64
    snpx_c3d.Convolution(name='Conv1a', num_filter=64, kernel=(3,3,3), pad=(1,1,1))
    snpx_c3d.MaxPool(name='Pool1', kernel=(1,2,2), stride=(1,2,2))

    # CONV2 3x3x3/1,128 
    snpx_c3d.Convolution(name='Conv2a', num_filter=128, kernel=(3,3,3), pad=(1,1,1))
    snpx_c3d.MaxPool(name='Pool2', kernel=(2,2,2), stride=(2,2,2))

    # CONV3 3x3x3/1,256 
    snpx_c3d.Convolution(name='Conv3a', num_filter=256, kernel=(3,3,3), pad=(1,1,1))
    snpx_c3d.Convolution(name='Conv3b', num_filter=256, kernel=(3,3,3), pad=(1,1,1))
    snpx_c3d.MaxPool(name='Pool3', kernel=(2,2,2), stride=(2,2,2))

    # CONV4 3x3x3/1,512 
    snpx_c3d.Convolution(name='Conv4a', num_filter=512, kernel=(3,3,3), pad=(1,1,1))
    snpx_c3d.Convolution(name='Conv4b', num_filter=512, kernel=(3,3,3), pad=(1,1,1))
    snpx_c3d.MaxPool(name='Pool4', kernel=(2,2,2), stride=(2,2,2))

    # CONV5 3x3x3/1,512 
    snpx_c3d.Convolution(name='Conv5a', num_filter=512, kernel=(3,3,3), pad=(1,1,1))
    snpx_c3d.Convolution(name='Conv5b', num_filter=512, kernel=(3,3,3), pad=(1,1,1))
    snpx_c3d.ZeroPadding(pad_size=(0,1,1))
    snpx_c3d.MaxPool(name='Pool5', kernel=(2,2,2), stride=(2,2,2))

    # Fully Connected
    snpx_c3d.Flatten()
    snpx_c3d.FullyConnected(num_hidden=4096, name="fc6", act='relu')
    snpx_c3d.Dropout()
    snpx_c3d.FullyConnected(num_hidden=4096, name="fc7", act='relu')
    snpx_c3d.Dropout()

    # Softmax Classifier
    snpx_c3d.Softmax(num_classes=num_classes)
    return snpx_c3d.Net