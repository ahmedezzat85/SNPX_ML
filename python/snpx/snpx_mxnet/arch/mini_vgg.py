import mxnet as mx
import numpy as np
from .mx_net import MxNet

def mini_vgg(net, use_bn=False):
    # CONVOLUTION_1 3x3/1,64
    net.convolution(64, (3,3), add_bn=use_bn, name='conv1_a')
    net.convolution(64, (3,3), add_bn=use_bn, name='conv1_b')
    net.pooling('max', (2,2), name='pool1')
    
    # CONVOLUTION_2 3x3/1,128 
    net.convolution(128, (3,3), add_bn=use_bn, name='conv2_a')
    net.convolution(128, (3,3), add_bn=use_bn, name='conv2_b')
    net.pooling('max', (2,2), name='pool2')

    # CONVOLUTION_3 3x3/1,256
    net.convolution(256, (3,3), add_bn=use_bn, name='conv3_a')
    net.convolution(256, (3,3), add_bn=use_bn, name='conv3_b')
    net.global_pool(pool_type='avg', name="global_pooling")
    return net

def snpx_net_create(num_classes, 
                    dtype=np.float32,
                    is_training=True,
                    use_bn=True):
    """ """
    net = MxNet(dtype, is_training)
    net = mini_vgg(net, use_bn)
    net.Softmax(num_classes, fc=False)
    return net.mx_sym