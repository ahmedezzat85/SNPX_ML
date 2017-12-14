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
        self.out_tensor  = input_data
        self.bias_init   = bias_init
        self.kernel_init = kernel_init
        self.data_format = data_format
        self.training    = train
        self.trainable   = train
        if data_format.startswith("NC"):
            self.channels_order = 'channels_first'
        else:
            self.channels_order = 'channels_last'

    def batch_norm(self, act=None, name=None):
        """ """
        bn_axis = -1 if self.channels_order == 'channels_last' else 1
        self.out_tensor = tf.layers.batch_normalization(
            inputs=self.out_tensor, 
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
            self.out_tensor = act(self.out_tensor)
        return self.out_tensor

    def convolution(self, num_filters, kernel, stride=(1,1), pad='same', act_fn='relu',
                    is_batch_norm=False, name=None):
        """ """
        # Activation
        if act_fn.lower() == 'leaky':
            act = tf_leaky_relu
        elif act_fn.lower() == 'relu':
            act = tf.nn.relu
        else:
            act = None
        conv_act = None if is_batch_norm == True else act
        
        kernel_init = self.kernel_init
        bias_init   = self.bias_init
        
        self.out_tensor = tf.layers.conv2d(inputs=self.out_tensor,
                                            filters=num_filters,
                                            kernel_size=kernel,
                                            strides=stride,
                                            padding=pad,
                                            data_format=self.channels_order,
                                            activation=conv_act,
                                            kernel_initializer=kernel_init,
                                            bias_initializer=bias_init,
                                            use_bias=False,
                                            trainable=self.trainable,
                                            name=name)
        if is_batch_norm == True:
            bn_name = None if name is None else name+'_BN'
            self.out_tensor = self.batch_norm(act=act, name=bn_name)
        return self.out_tensor
      
    def conv_dw(self, num_filters=None, kernel=(3,3), stride=(1,1), pad='same', act_fn='relu',
                    is_batch_norm=False, name=None):
        """ """
        # Activation
        if act_fn.lower() == 'leaky':
            act = tf_leaky_relu
        elif act_fn.lower() == 'relu':
            act = tf.nn.relu
        else:
            act = None
        conv_act = None if is_batch_norm == True else act
        
        self.out_tensor = separable_conv2d(inputs=self.out_tensor,
                                            num_outputs=None,
                                            kernel_size=kernel,
                                            depth_multiplier=1,
                                            stride=stride,
                                            padding='SAME',
                                            data_format=self.data_format,
                                            trainable=self.trainable,
                                            activation_fn=conv_act,
                                            scope=name+'_DW')
        if is_batch_norm == True:
            bn_name = None if name is None else name+'DW_BN'
            self.out_tensor = self.batch_norm(act=act, name=bn_name)

        # PointWise Convolution
        if num_filters is not None:
            self.out_tensor = self.convolution(num_filters=num_filters, 
                                                kernel=(1,1),
                                                act_fn=act_fn,
                                                is_batch_norm=is_batch_norm,
                                                name=name)
        return self.out_tensor
         
    def pooling(self, pool_type, kernel, stride=None, name=None):
        """ """
        if stride == None:
            stride = kernel
        if pool_type.lower() == 'max':
            self.out_tensor = tf.layers.max_pooling2d(inputs=self.out_tensor,
                                                        pool_size=kernel,
                                                        strides=stride,
                                                        data_format=self.channels_order,
                                                        name=name)
        elif pool_type.lower() == 'avg':
            self.out_tensor = tf.layers.average_pooling2d(inputs=self.out_tensor,
                                                            pool_size=kernel,
                                                            strides=stride,
                                                            data_format=self.channels_order,
                                                            name=name)            
        return self.out_tensor
    def flatten(self):
        """ """
        self.out_tensor = flatten(self.out_tensor)

    def fully_connected(self, units, act_fn='', name=None):
        """ """
        if act_fn.lower() == 'leaky':
            act = tf_leaky_relu
        elif act_fn.lower() == 'relu':
            act = tf.nn.relu
        else:
            act = None
        self.out_tensor = tf.layers.dense(
            inputs=self.out_tensor,
            units=units,
            activation=act,
            kernel_initializer=self.kernel_init,
            bias_initializer=self.bias_init,
            trainable=self.trainable,
            name=name)
        return self.out_tensor

