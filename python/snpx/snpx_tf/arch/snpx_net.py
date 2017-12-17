from __future__ import absolute_import

import tensorflow as tf
from . tf_slim_net import TFNet, dropout
from .. import tf_train_utils as tf_train

def model_A(net):
    net.convolution(16, (3,3), stride=(1,1), name='Conv0')
    net.conv_dw(16 , (3,3), stride=(1,1), add_batch_norm=True, name='Conv1')
    net.conv_dw(32 , (3,3), stride=(2,2), add_batch_norm=True, name='Conv2')
    net.conv_dw(32 , (3,3), stride=(1,1), add_batch_norm=True, name='Conv3')
    net.conv_dw(32 , (3,3), stride=(1,1), add_batch_norm=True, name='Conv4')
    net.conv_dw(64 , (3,3), stride=(2,2), add_batch_norm=True, name='Conv5')
    net.conv_dw(64 , (3,3), stride=(1,1), add_batch_norm=True, name='Conv6')
    net.conv_dw(64 , (3,3), stride=(1,1), add_batch_norm=True, name='Conv7')
    net.conv_dw(64 , (3,3), stride=(1,1), add_batch_norm=True, name='Conv8')
    net.conv_dw(64 , (3,3), stride=(1,1), add_batch_norm=True, name='Conv9')
    net.conv_dw(128, (3,3), stride=(2,2), add_batch_norm=True, name='Conv10')
    net.pooling('avg', (4,4), name="global_pool")
    return net

def model_B(net):
    net.convolution(16, (3,3), stride=(1,1), name='Conv0')
    net.conv_dw(16 , (3,3), stride=(1,1), add_batch_norm=True, name='Conv1')
    net.conv_dw(32 , (3,3), stride=(2,2), add_batch_norm=True, name='Conv2')
    net.conv_dw(32 , (3,3), stride=(1,1), add_batch_norm=True, name='Conv3')
    net.conv_dw(32 , (3,3), stride=(1,1), add_batch_norm=True, name='Conv4')
    net.conv_dw(64 , (3,3), stride=(2,2), add_batch_norm=True, name='Conv5')
    net.conv_dw(64 , (3,3), stride=(1,1), add_batch_norm=True, name='Conv6')
    net.conv_dw(64 , (3,3), stride=(1,1), add_batch_norm=True, name='Conv7')
    net.conv_dw(64 , (3,3), stride=(1,1), add_batch_norm=True, name='Conv8')
    net.conv_dw(64 , (3,3), stride=(1,1), add_batch_norm=True, name='Conv9')
    net.pooling('avg', (8,8), name="global_pool")
    return net

def snpx_net_create(num_classes, 
                    input_data,
                    data_format="NHWC",
                    is_training=True):
    """ """
    dtype = input_data.dtype.base_dtype
    
    #CONVOLUTION_1 3x3/1,64
    net = TFNet(input_data, data_format, train=is_training)
    net = model_B(net)
    net.out_tensor = dropout(net.out_tensor)
    net.flatten()
    net.fully_connected(num_classes, name='FC_Softmax')
    return net.out_tensor
