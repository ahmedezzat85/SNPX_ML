from __future__ import absolute_import

import mxnet as mx
import numpy as np
from .mx_net import MxNet


class MiniVGG(MxNet):
    """
    """
    def __init__(self, data_format, num_classes, is_train=True):
        super(MiniVGG, self).__init__(train=is_train)
        self.num_classes = num_classes

    def __call__(self, blocks, filters=[32, 64, 128], strides=[2,2,1],bn=True):
        net_out = self.net_out
        for i in range(len(blocks)):
            f = filters[i]
            n = blocks[i]
            for k in range(n):
                net_out = self.convolution(net_out, f, (3,3), add_bn=bn, name='Conv'+str(i+1)+str(k+1))
            if strides[i] > 1:
                net_out = self.pooling(net_out, 'max', (2,2), name='Pool'+str(i+1))

        net_out = self.global_pool(net_out, 'avg', name='global_pool')
        net_out = self.dropout(net_out, 0.5)
        net_out = self.flatten(net_out)
        net_out = self.Softmax(net_out, self.num_classes)
        return net_out

def snpx_net_create(num_classes, data_format="NHWC", is_training=True):
    """ """
    net = MiniVGG(data_format, num_classes, is_training)
    net_out = net(blocks=[2, 2, 2], filters=[64, 128, 256], strides=[2,2,1])
    # net_out = net(blocks=[7, 6, 6], filters=[16, 32, 64], strides=[2,2,1], bn=False)
    return net_out
