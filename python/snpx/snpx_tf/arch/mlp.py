from __future__ import absolute_import

import tensorflow as tf
from . tf_net import TFNet
from .. import tf_train_utils as tf_train

def snpx_net_create(num_classes, 
                    input_data,
                    data_format="NHWC",
                    is_training=True):
    """ """
    dtype = input_data.dtype.base_dtype
    
    #CONVOLUTION_1 3x3/1,64
    net = TFNet(input_data, data_format, train=is_training,
                kernel_init=tf_train.xavier_initializer(dtype=dtype), 
                bias_init=tf.zeros_initializer(dtype))
    
    net.flatten()
    net.fully_connected(256, act_fn='relu', name='FC1')
    net.fully_connected(num_classes, name='FC_Softmax')
    return net.out_tensor