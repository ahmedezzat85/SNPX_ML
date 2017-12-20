from __future__ import absolute_import

import tensorflow as tf
from . tf_net import TFNet
from .. import tf_train_utils as tf_train

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
    net.pooling('avg', (8,8), name="global_pool")
    return net

def snpx_net_create(num_classes, 
                    input_data,
                    data_format="NHWC",
                    is_training=True,
                    use_bn=True):
    """ """
    dtype = input_data.dtype.base_dtype
    
    net = TFNet(input_data, data_format, train=is_training,
                kernel_init=tf_train.xavier_initializer(dtype=dtype), 
                bias_init=tf.zeros_initializer(dtype))
    net = mini_vgg(net, use_bn)
    net.convolution(num_classes, (1,1), pad='valid', act_fn='', name='Conv_Softmax')
    net.flatten()
    return net.out_tensor