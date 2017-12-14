from __future__ import absolute_import

import tensorflow as tf
from . tf_net import TFNet
from .. import tf_train_utils as tf_train
from . mobilenet_v1 import mobilenet_v1


def snpx_net_create00(num_classes, input_data, data_format='NHWC', is_training=True):
    """ """
    dtype   = input_data.dtype.base_dtype
    out, __ = mobilenet_v1(input_data, num_classes, prediction_fn=None, is_training=is_training)
    return out

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
    net.convolution(32, (3,3), stride=(2,2), name='Conv0')
    net.conv_dw(64, (3,3), stride=(1,1), is_batch_norm=True, name='Conv1')
    net.conv_dw(128, (3,3), stride=(2,2), is_batch_norm=True, name='Conv2')
    net.conv_dw(128, (3,3), stride=(1,1), is_batch_norm=True, name='Conv3')
    net.conv_dw(256, (3,3), stride=(2,2), is_batch_norm=True, name='Conv4')
    net.conv_dw(256, (3,3), stride=(1,1), is_batch_norm=True, name='Conv5')
    net.conv_dw(512, (3,3), stride=(2,2), is_batch_norm=True, name='Conv6')
    net.conv_dw(512, (3,3), stride=(1,1), is_batch_norm=True, name='Conv7')
    net.conv_dw(512, (3,3), stride=(1,1), is_batch_norm=True, name='Conv8')
    net.conv_dw(512, (3,3), stride=(1,1), is_batch_norm=True, name='Conv9')
    net.conv_dw(512, (3,3), stride=(1,1), is_batch_norm=True, name='Conv10')
    net.conv_dw(512, (3,3), stride=(1,1), is_batch_norm=True, name='Conv11')
    net.conv_dw(1024, (3,3), stride=(2,2), is_batch_norm=True, name='Conv12')
    net.conv_dw(1024, (3,3), stride=(2,2), is_batch_norm=True, name='Conv13')
    # net.pooling('avg', (7,7), name="global_pool")
    net.flatten()
    net.fully_connected(num_classes, name='FC_Softmax')
    return net.out_tensor
