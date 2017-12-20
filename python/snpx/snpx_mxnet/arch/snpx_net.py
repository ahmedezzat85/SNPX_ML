from __future__ import absolute_import

import mxnet as mx
import numpy as np
from .mx_net import MxNet

def model_A(net):
    net.convolution(16, (3,3), stride=(1,1), name='Conv0')
    net.conv_dw(16 , (3,3), stride=(1,1), add_bn=True, name='Conv1')
    net.conv_dw(32 , (3,3), stride=(2,2), add_bn=True, name='Conv2')
    net.conv_dw(32 , (3,3), stride=(1,1), add_bn=True, name='Conv3')
    net.conv_dw(32 , (3,3), stride=(1,1), add_bn=True, name='Conv4')
    net.conv_dw(64 , (3,3), stride=(2,2), add_bn=True, name='Conv5')
    net.conv_dw(64 , (3,3), stride=(1,1), add_bn=True, name='Conv6')
    net.conv_dw(64 , (3,3), stride=(1,1), add_bn=True, name='Conv7')
    net.conv_dw(64 , (3,3), stride=(1,1), add_bn=True, name='Conv8')
    net.conv_dw(64 , (3,3), stride=(1,1), add_bn=True, name='Conv9')
    net.conv_dw(128, (3,3), stride=(2,2), add_bn=True, name='Conv10')
    net.pooling('avg', (4,4), name="global_pool")
    return net

def model_B(net):
    net.convolution(16, (3,3), stride=(1,1), name='Conv0')
    net.conv_dw(16 , (3,3), stride=(1,1), add_bn=True, name='Conv1')
    net.conv_dw(32 , (3,3), stride=(2,2), add_bn=True, name='Conv2')
    net.conv_dw(32 , (3,3), stride=(1,1), add_bn=True, name='Conv3')
    net.conv_dw(32 , (3,3), stride=(1,1), add_bn=True, name='Conv4')
    net.conv_dw(64 , (3,3), stride=(2,2), add_bn=True, name='Conv5')
    net.conv_dw(64 , (3,3), stride=(1,1), add_bn=True, name='Conv6')
    net.conv_dw(64 , (3,3), stride=(1,1), add_bn=True, name='Conv7')
    net.conv_dw(64 , (3,3), stride=(1,1), add_bn=True, name='Conv8')
    net.conv_dw(64 , (3,3), stride=(1,1), add_bn=True, name='Conv9')
    net.pooling('avg', (8,8), name="global_pool")
    return net

def model_C(net):
    net.convolution(32, (3,3), stride=(1,1), name='Conv0')
    net.conv_dw(32  , (3,3), stride=(1,1), add_bn=True, name='Conv1')
    net.conv_dw(64  , (3,3), stride=(2,2), add_bn=True, name='Conv2')
    net.conv_dw(64  , (3,3), stride=(1,1), add_bn=True, name='Conv3')
    net.conv_dw(64  , (3,3), stride=(1,1), add_bn=True, name='Conv4')
    net.conv_dw(128 , (3,3), stride=(2,2), add_bn=True, name='Conv5')
    net.conv_dw(128 , (3,3), stride=(1,1), add_bn=True, name='Conv6')
    net.conv_dw(128  , (3,3), stride=(1,1), add_bn=True, name='Conv7')
    net.conv_dw(128  , (3,3), stride=(1,1), add_bn=True, name='Conv8')
    net.conv_dw(128  , (3,3), stride=(1,1), add_bn=True, name='Conv9')
    net.pooling('avg', (8,8), name="global_pool")
    return net

def snpx_net_create(num_classes, 
                    dtype=np.float32,
                    is_training=True,
                    use_bn=True):
    """ """
    net = MxNet(dtype, is_training)
    net = model_C(net)
    net.convolution(num_classes, (1,1), pad='valid', act_fn='', name='Conv_Softmax')
    net.flatten()
    net.Softmax(num_classes)
    return net.mx_sym