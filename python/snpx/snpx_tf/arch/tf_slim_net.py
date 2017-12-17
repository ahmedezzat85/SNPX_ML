from __future__ import absolute_import

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers import (
    flatten, separable_conv2d, batch_norm, conv2d, fully_connected, dropout
)
TF_ACT_FN = {'relu': tf.nn.relu}

class TFNet(object):
    """
    """
    def __init__(self, input_data, data_format='NHWC', kernel_init=None, bias_init=None, train=True):
        self.dtype       = input_data.dtype.base_dtype
        self.out_tensor  = input_data
        self.bias_init   = bias_init
        self.kernel_init = kernel_init
        self.data_format = data_format
        self.trainable   = train

    def batch_norm(self, act=None, name=None):
        """ """
        self.out_tensor = batch_norm(inputs=self.out_tensor, 
                                     activation_fn=act,
                                     is_training=True,
                                     trainable=self.trainable,
                                     fused=True,
                                     data_format=self.data_format,
                                     scope=name)
        return self.out_tensor

    def convolution(self, num_filters, kernel, stride=(1,1), pad='same', act_fn='relu',
                    add_batch_norm=False, name=None):
        """ """
        # Activation
        act = None
        if act_fn in TF_ACT_FN:
            act = TF_ACT_FN[act_fn]
        if add_batch_norm == True:
            conv_act = None
            bn_name = name+'_BN'
            bn_fn=batch_norm
            bn_fn_args={'activation_fn': act, 'is_training': True, 'trainable': self.trainable,
                        'fused': True, 'data_format': self.data_format, 'scope': bn_name}
        else:
            conv_act = act
            bn_fn=None
            bn_fn_args=None
        
        self.out_tensor = conv2d(inputs=self.out_tensor,
                                 num_outputs=num_filters,
                                 kernel_size=kernel,
                                 stride=list(stride),
                                 padding=pad.upper(),
                                 data_format=self.data_format,
                                 activation_fn=conv_act,
                                 normalizer_fn=bn_fn,
                                 normalizer_params=bn_fn_args,
                                 trainable=self.trainable,
                                 scope=name)
        return self.out_tensor
      
    def conv_dw(self, num_filters=None, kernel=(3,3), stride=(1,1), pad='same', act_fn='relu',
                    add_batch_norm=False, name=None):
        """ """
        # Activation
        dw_name = name + '_dw'
        act = None
        if act_fn in TF_ACT_FN:
            act = TF_ACT_FN[act_fn]
        
        if add_batch_norm == True:
            conv_act = None
            bn_name  = dw_name + '_BN'
            bn_fn    = batch_norm
            dw_bn_fn_args = {'activation_fn': act, 'is_training': True, 'trainable': self.trainable,
                             'fused': True, 'data_format': self.data_format, 'scope': bn_name}
            bn_fn_args = dw_bn_fn_args
            bn_fn_args['scope'] = name + '_BN'
        else:
            conv_act = act
            bn_fn=None
            bn_fn_args=None

        self.out_tensor = separable_conv2d(inputs=self.out_tensor,
                                           num_outputs=None,
                                           kernel_size=kernel,
                                           depth_multiplier=1,
                                           stride=list(stride),
                                           padding=pad.upper(),
                                           data_format=self.data_format,
                                           trainable=self.trainable,
                                           activation_fn=conv_act,
                                           normalizer_fn=bn_fn,
                                           normalizer_params=dw_bn_fn_args,
                                           scope=dw_name)
        # PointWise Convolution
        if num_filters is not None:
            self.convolution(num_filters=num_filters, 
                             kernel=(1,1),
                             act_fn=act_fn,
                             add_batch_norm=add_batch_norm,
                             name=name)
        return self.out_tensor
         
    def pooling(self, pool_type, kernel, stride=None, name=None):
        """ """
        if stride == None:
            stride = kernel
        self.out_tensor = slim.pool(inputs=self.out_tensor,
                                kernel_size=kernel,
                                pooling_type=pool_type.upper(),
                                data_format=self.data_format,
                                scope=name)
        return self.out_tensor

    def flatten(self):
        """ """
        self.out_tensor = flatten(self.out_tensor)

    def fully_connected(self, units, act_fn='', name=None):
        """ """
        act = None
        if act_fn in TF_ACT_FN:
            act = TF_ACT_FN[act_fn]

        self.out_tensor = fully_connected(
            inputs=self.out_tensor,
            num_outputs=units,
            activation_fn=act,
            trainable=self.trainable,
            scope=name
        )
        return self.out_tensor

