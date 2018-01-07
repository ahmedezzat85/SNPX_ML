from __future__ import absolute_import

import mxnet as mx
import numpy as np

class MxNet(object):
    """
    """
    def __init__(self, dtype='float32', train=True):
        self.dtype     = dtype
        self.training  = train
        self.cudnn     = 'fastest'
        self.net_out   = mx.sym.var(name='data', dtype=dtype)
        if dtype == 'float16':
            self.net_out   = mx.sym.Cast(data=self.net_out, dtype=np.float16)

    def batch_norm(self, data, act_fn='relu', name=None):
        """ """
        net_out = mx.sym.BatchNorm(data=data)
        if act_fn:
            net_out = mx.sym.Activation(data=net_out, act_type=act_fn)
        return net_out

    def convolution(self,
                    data,
                    num_filters,
                    kernel,
                    stride=(1,1),
                    pad='same',
                    act_fn='relu',
                    add_bn=False,
                    no_bias=False,
                    name=None):
        """ """
        padding = (1,1) if pad.lower() == 'same' else (0,0)

        # Convolution
        net_out = mx.sym.Convolution(data=data, kernel=kernel, stride=stride, pad=padding,
                                     num_filter=num_filters, cudnn_tune=self.cudnn,
                                     no_bias=(no_bias or add_bn), name=name)
        
        # Batch Normalization
        if add_bn == True:
            bn_name = None if name is None else name+'_BN'
            net_out = mx.sym.BatchNorm(data=net_out, name=bn_name)
        
        # Activation
        if act_fn:
            net_out = mx.sym.Activation(data=net_out, act_type=act_fn, name=name +'_'+act_fn)
        return net_out
      
    def conv_dw(self, num_filters, kernel=(3,3), stride=(1,1), pad='same', act_fn='relu',
                    add_bn=False, ptwise_conv=True, inputs=None, name=None):
        """ """
        dw_name = name + '_dw' if name is not None else None
        padding = (1,1) if pad.lower() == 'same' else (0,0)
        if stride == (1,1):
            dw_filters = num_filters
        elif stride == (2,2):
            dw_filters = num_filters // 2

        # Convolution
        data = self.net_out if inputs is None else inputs
        self.net_out = mx.sym.Convolution(data=data, 
                                         no_bias=add_bn,
                                         kernel=kernel,
                                         stride=stride,
                                         pad=padding,
                                         num_filter=dw_filters,
                                         num_group=dw_filters,
                                         cudnn_tune=self.cudnn,
                                         name=dw_name)
        
        # Batch Normalization
        if add_bn == True:
            bn_name = None if name is None else dw_name+'_BN'
            self.net_out = mx.sym.BatchNorm(data=self.net_out, name=bn_name)
        
        # Activation
        self.net_out = mx.sym.Activation(data=self.net_out,
                                        act_type=act_fn,
                                        name=name+'_'+act_fn)
        
        # PointWise Convolution
        if ptwise_conv is True:
            net_out = self.convolution(num_filters, (1,1), stride=(1,1), pad='same', act_fn=act_fn,
                                add_bn=add_bn, name=name)
        return net_out
         
    def pooling(self, data, pool_type, kernel, stride=None, name=None):
        """ """
        if stride is None:
            stride = kernel
        net_out = mx.sym.Pooling(data=data, kernel=kernel, pool_type=pool_type,
                                    stride=stride, name=name)
        return net_out

    def global_pool(self, data, pool_type='avg', name=None):
        """ """
        net_out = mx.sym.Pooling(data, global_pool=True, kernel=(1,1), pool_type=pool_type, name=name)
        return net_out
        
    def flatten(self, data):
        """ """
        net_out = mx.sym.Flatten(data, name='flatten')
        return net_out

    def fully_connected(self, data, units, add_bn=False, act_fn='', name=None):
        """ """
        net_out = mx.sym.FullyConnected(data, num_hidden=units, name=name)

        if add_bn == True:
            bn_name = None if name is None else name+'_BN'
            net_out = mx.sym.BatchNorm(data=net_out, name=bn_name)

        if act_fn:
            net_out = mx.sym.Activation(data, act_type=act_fn, name=name + '_' + act_fn)
        return net_out

    def Softmax(self, data, num_classes, fc=True):
        """
        """
        if fc == False:
            net_out = self.convolution(data, num_classes, (1,1), pad='valid', act_fn='', 
                                        name='Conv_Softmax')
            self.flatten(net_out)
        else:
            net_out = self.fully_connected(data, num_classes, name="FC_softmax")
        if self.dtype == 'float16':
            net_out = mx.sym.Cast(net_out, dtype=np.float32)
        net_out = mx.sym.SoftmaxOutput(net_out, name='softmax')
        return net_out
