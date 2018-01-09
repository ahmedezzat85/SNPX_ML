from __future__ import absolute_import

import tensorflow as tf
from . tf_net import TFNet

class MLP(TFNet):
    """
    """
    def __init__(self, data, data_format, num_classes, is_train=True):
        dtype = data.dtype.base_dtype
        super(MLP, self).__init__(dtype, data_format, train=is_train)
        self.net_out = tf.identity(data, name='data')
        self.num_classes = num_classes

    def __call__(self, hidden=2):
        net_out = self.flatten(self.net_out)
        for k in range(hidden):
            net_out = self.fully_connected(net_out, 128, add_bn=True, act_fn='relu', name='fc_'+str(k))
        
        net_out = self.Softmax(net_out, self.num_classes)
        return net_out

def snpx_net_create(num_classes, input_data, data_format="NHWC", is_training=True):
    """ """
    net = MLP(input_data, data_format, num_classes, is_training)
    net_out = net()
    return net_out
