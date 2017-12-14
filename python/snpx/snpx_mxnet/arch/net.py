""" Neural Network Operations
"""
from __future__ import absolute_import

import mxnet as mx
import numpy as np

class SNPXNet(object):
    def __init__(self, fp16=False, batch_norm=False):
        self.fp16       = fp16
        self.cudnn      = 'fastest'
        self.layer_cnt  = 0
        self.batch_norm = batch_norm

        # Add Input Data
        self.Net = mx.sym.var('data')
        if self.fp16 == True:
            self.Net = mx.sym.Cast(data=self.Net, dtype=np.float16)

    def Convolution(self, num_filter, kernel, data=None, act='relu', stride=None, pad=(), incr=True, name=None):
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
        if stride is None:
            if len(kernel) == 2:
                stride = (1,1)
            elif len(kernel) == 3:
                stride = (1,1,1)
            else:
                raise ValueError("Convolution Kernel Cannot be larger than 3")
        
        if data is None:
            data = self.Net
        
        if name is None:
            name = "CONV_" + str(self.layer_cnt)

        # Create Weight Variables
        weight  = mx.sym.var(name=name+'_weight', dtype=np.float32)
        bias    = mx.sym.var(name=name+'_bias'  , dtype=np.float32)
        if self.fp16==True:
            weight  = mx.sym.Cast(data=weight   , dtype=np.float16)
            bias    = mx.sym.Cast(data=bias     , dtype=np.float16)
        
        # Convolution
        Conv = mx.sym.Convolution(data=data, num_filter=num_filter, name=name, kernel=kernel, pad=pad, 
                                    stride=stride, cudnn_tune=self.cudnn, weight=weight, bias=bias)
        # Add Batch Normalization Block (if configured)
        if self.batch_norm == True:
            Conv = mx.sym.BatchNorm(data=Conv)

        # Non-Linearity
        Conv = mx.sym.Activation(data=Conv, act_type=act)

        # Add the block sequentially to the network
        if incr == True:
            self.Net = Conv

        self.layer_cnt += 1
        return Conv

    def FullyConnected(self, num_hidden, name, data=None, act=None):
        """
        """
        if data is None:
            data = self.Net
        
        if name is None:
            name = "FC_" + str(self.layer_cnt)
        self.layer_cnt += 1

        weight  = mx.sym.var(name=name+'_weight', dtype=np.float32)
        bias    = mx.sym.var(name=name+'_bias'  , dtype=np.float32)
        if self.fp16 == True:
            weight  = mx.sym.Cast(data=weight   , dtype=np.float16)
            bias    = mx.sym.Cast(data=bias     , dtype=np.float16)
        
        # Fully-Connected Layer
        FC = mx.sym.FullyConnected(data=data, num_hidden=num_hidden, name=name, weight=weight, bias=bias)

        # Add Batch Normalization Block (if configured)
        if self.batch_norm == True:
            FC = mx.sym.BatchNorm(data=FC)
        
        # Activation
        if act is not None:
            self.Net = mx.sym.Activation(data=FC, act_type=act)
        else:
            self.Net = FC

    def Softmax(self, num_classes):
        """
        """
        self.FullyConnected(num_hidden=num_classes, name="FC_softmax")
        if self.fp16 == True:
            label = mx.sym.var(name='softmax_label')
            label = mx.sym.Cast(data=label, dtype=np.float16)
            out   = mx.sym.SoftmaxOutput(data=self.Net, name='softmax', label=label)
        else:
            out   = mx.sym.SoftmaxOutput(data=self.Net, name='softmax')
        self.Net = out

    def Flatten(self):
        self.Net = mx.sym.Flatten(data=self.Net)
    
    def MaxPool(self, name, kernel, stride):
        self.Net = mx.sym.Pooling(data=self.Net, name=name, pool_type='max', kernel=kernel, stride=stride)

    def AvgPool(self, name, kernel, stride):
        self.Net = mx.sym.Pooling(data=self.Net, name=name, pool_type='avg', kernel=kernel, stride=stride)

    def Dropout(self):
        self.Net = mx.sym.Dropout(data=self.Net)

    def ZeroPadding(self, pad_size):
        pad_size = (0,0,) + pad_size
        pad = np.array([pad_size, pad_size])
        pad = pad.reshape(len(pad_size) * 2, order='F')
        self.Net = mx.sym.Pad(data=self.Net, mode='constant', pad_width=tuple(pad))

    def Concat(self, L=[]):
        self.Net = mx.sym.Concat(*L)