from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib.layers import flatten, separable_conv2d

def tf_leaky_relu(x, name=None):
    return tf.nn.relu(x) - 0.1 * tf.nn.relu(-x) # Memory consuming implementation

class TFNet(object):
    """
    """
    def __init__(self, input_data, data_format='NHWC', kernel_init=None, bias_init=None, train=True):
        self.dtype       = input_data.dtype.base_dtype
        self.net_out  = input_data
        self.bias_init   = bias_init
        self.kernel_init = kernel_init
        self.data_format = data_format
        self.training    = train
        self.trainable   = train
        if data_format.startswith("NC"):
            self.channels_order = 'channels_first'
        else:
            self.channels_order = 'channels_last'

    def batch_norm(self, inputs=None, act_fn='relu', name=None):
        """ """
        # Activation
        if act_fn == 'relu':
            act = tf.nn.relu
        else:
            act = None

        bn_axis = -1 if self.channels_order == 'channels_last' else 1
        data = self.net_out if inputs is None else inputs
        self.net_out = tf.layers.batch_normalization(
            inputs=data, 
            axis=bn_axis, 
            scale=False,
            beta_initializer=tf.zeros_initializer(self.dtype),
            gamma_initializer=tf.zeros_initializer(self.dtype),
            moving_mean_initializer=tf.zeros_initializer(self.dtype),
            moving_variance_initializer=tf.ones_initializer(self.dtype),
            training=True,
            trainable=self.trainable,
            name=name,
            fused=True)
        if act is not None:
            self.net_out = act(self.net_out)
        return self.net_out

    def convolution(self, num_filters, kernel, stride=(1,1), pad='same', act_fn='relu',
                    add_bn=False, name=None, inputs=None, no_bias=False):
        """ """
        # Activation
        if act_fn.lower() == 'leaky':
            act = tf_leaky_relu
        elif act_fn.lower() == 'relu':
            act = tf.nn.relu
        else:
            act = None
        conv_act = None if add_bn == True else act
        
        kernel_init = self.kernel_init
        bias_init   = self.bias_init
        data = self.net_out if inputs is None else inputs
        self.net_out = tf.layers.conv2d(inputs=data,
                                            filters=num_filters,
                                            kernel_size=kernel,
                                            strides=stride,
                                            padding=pad,
                                            data_format=self.channels_order,
                                            activation=conv_act,
                                            kernel_initializer=kernel_init,
                                            bias_initializer=bias_init,
                                            use_bias=not no_bias,
                                            trainable=self.trainable,
                                            name=name)
        if add_bn == True:
            bn_name = None if name is None else name+'_BN'
            self.net_out = self.batch_norm(act_fn=act_fn, name=bn_name)
        return self.net_out
      
    def conv_dw(self, num_filters=None, kernel=(3,3), stride=(1,1), pad='same', act_fn='relu',
                    add_bn=False, name=None):
        """ """
        # Activation
        if act_fn.lower() == 'leaky':
            act = tf_leaky_relu
        elif act_fn.lower() == 'relu':
            act = tf.nn.relu
        else:
            act = None
        conv_act = None if add_bn == True else act
        
        self.net_out = separable_conv2d(inputs=self.net_out,
                                            num_outputs=None,
                                            kernel_size=kernel,
                                            depth_multiplier=1,
                                            stride=stride,
                                            padding='SAME',
                                            data_format=self.data_format,
                                            trainable=self.trainable,
                                            activation_fn=conv_act,
                                            scope=name+'_DW')
        if add_bn == True:
            bn_name = None if name is None else name+'DW_BN'
            self.net_out = self.batch_norm(act_fn=act, name=bn_name)

        # PointWise Convolution
        if num_filters is not None:
            self.net_out = self.convolution(num_filters=num_filters, 
                                                kernel=(1,1),
                                                act_fn=act_fn,
                                                add_bn=add_bn,
                                                name=name)
        return self.net_out
         
    def pooling(self, pool_type, kernel, stride=None, name=None):
        """ """
        if stride == None:
            stride = kernel
        if pool_type.lower() == 'max':
            self.net_out = tf.layers.max_pooling2d(inputs=self.net_out,
                                                        pool_size=kernel,
                                                        strides=stride,
                                                        data_format=self.channels_order,
                                                        name=name)
        elif pool_type.lower() == 'avg':
            self.net_out = tf.layers.average_pooling2d(inputs=self.net_out,
                                                            pool_size=kernel,
                                                            strides=stride,
                                                            data_format=self.channels_order,
                                                            name=name)            
        return self.net_out
    def flatten(self):
        """ """
        self.net_out = flatten(self.net_out)

    def fully_connected(self, units, act_fn='', name=None):
        """ """
        if act_fn.lower() == 'leaky':
            act = tf_leaky_relu
        elif act_fn.lower() == 'relu':
            act = tf.nn.relu
        else:
            act = None
        self.net_out = tf.layers.dense(
            inputs=self.net_out,
            units=units,
            activation=act,
            kernel_initializer=self.kernel_init,
            bias_initializer=self.bias_init,
            trainable=self.trainable,
            name=name)
        return self.net_out