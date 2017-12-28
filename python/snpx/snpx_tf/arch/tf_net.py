from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib.layers import separable_conv2d, l2_regularizer

TF_ACT_FN = {'relu': tf.nn.relu, 'leaky': tf.nn.leaky_relu}

class TFNet(object):
    """
    """
    def __init__(self,
                 data_type=tf.float32, 
                 data_format='NHWC',
                 kernel_init=tf.variance_scaling_initializer(),
                 l2_reg=0,
                 train=True):
        self.dtype       = data_type
        self.kernel_init = kernel_init
        self.regulaizer  = None
        self.data_format = data_format
        self.trainable   = train
        self.bn_eps      = 1e-5
        self.bn_decay    = .997
        if data_format.startswith("NC"):
            self.channels_order = 'channels_first'
        else:
            self.channels_order = 'channels_last'

    def batch_norm(self, data, act_fn='relu', name=None):
        """ """
        # Batch Normalization Layer
        bn_axis = 1 if self.data_format.startswith('NC') else -1
        net_out = tf.layers.batch_normalization(inputs=data, axis=bn_axis, training=True,
                                                epsilon=self.bn_eps, momentum=self.bn_decay,
                                                trainable=self.trainable, fused=True, name=name)
        # Activation
        if act_fn in TF_ACT_FN:
            act = TF_ACT_FN[act_fn]
            net_out = act(net_out, name=name+'_'+act_fn)

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
        # Convolution with no activation
        net_out = tf.layers.conv2d(inputs=data,
                                    filters=num_filters,
                                    kernel_size=kernel,
                                    strides=stride,
                                    padding=pad,
                                    data_format=self.channels_order,
                                    activation=None,
                                    kernel_initializer=self.kernel_init,
                                    kernel_regularizer=self.regulaizer,
                                    use_bias=not (add_bn or no_bias),
                                    trainable=self.trainable,
                                    name=name)
        # Add Batch normalization if required
        if add_bn == True:
            bn_name = None if name is None else name+'_Bn'
            net_out = self.batch_norm(data=net_out, act_fn='', name=bn_name)

        # Activation
        if act_fn in TF_ACT_FN:
            act = TF_ACT_FN[act_fn]
            net_out = act(net_out, name=name+'_'+act_fn)

        return net_out
      
    def conv_dw(self,
                data,
                num_filters=None,
                kernel=(3,3),
                stride=(1,1),
                pad='same',
                act_fn='relu',
                add_bn=False,
                name=None):
        """ """
        net_out = separable_conv2d(inputs=data,
                                    num_outputs=None,
                                    kernel_size=kernel,
                                    depth_multiplier=1,
                                    stride=stride,
                                    padding='SAME',
                                    data_format=self.data_format,
                                    trainable=self.trainable,
                                    activation_fn=None,
                                    scope=name+'_dw')
        if add_bn == True:
            bn_name = None if name is None else name+'dw_Bn'
            net_out = self.batch_norm(net_out, act_fn='', name=bn_name)

        # Activation
        if act_fn in TF_ACT_FN:
            act = TF_ACT_FN[act_fn]
            net_out = act(net_out, name=name+act_fn)

        # PointWise Convolution
        if num_filters is not None:
            net_out = self.convolution(net_out, num_filters, (1,1), (1,1), 'same', act_fn,
                                            add_bn, name=name)
        return net_out
         
    def pooling(self, data, pool_type, kernel, stride=None, name=None):
        """ """
        if stride == None: stride = kernel
        if pool_type.lower() == 'max':
            net_out = tf.layers.max_pooling2d(data, kernel, stride,
                                              data_format=self.channels_order, name=name)
        elif pool_type.lower() == 'avg':
            net_out = tf.layers.average_pooling2d(data, kernel, stride,
                                                  data_format=self.channels_order, name=name)
        return net_out

    def flatten(self, data):
        """ """
        net_out = tf.layers.flatten(data, name='Flatten_0')
        return net_out

    def fully_connected(self, data, units, add_bn=False, act_fn='', name=None):
        """ """
        net_out = tf.layers.dense(data, units, activation=None, use_bias=not add_bn,
                                  kernel_initializer=self.kernel_init, trainable=self.trainable,
                                  kernel_regularizer=self.regulaizer, name=name)
        return net_out

    def Softmax(self, data, num_classes, fc=True):
        """
        """
        if fc == False:
            net_out = self.convolution(data, num_classes, (1,1), pad='valid', act_fn='', 
                                        name='Conv_Softmax')
            net_out = self.flatten(net_out)
        else:
            net_out = self.fully_connected(data, num_classes, name="FC_softmax")
        if self.dtype != tf.float32:
            net_out = tf.cast(net_out, tf.float32)
        predictions = tf.nn.softmax(net_out, name='Output')
        return net_out, predictions