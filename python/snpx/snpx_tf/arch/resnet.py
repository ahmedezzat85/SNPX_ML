import tensorflow as tf
from . tf_net import TFNet
from .. import tf_train_utils as tf_train

class Resnet(TFNet):
    """
    """
    def __init__(self, data, data_format, num_classes, is_train=True):
        super(Resnet, self).__init__(data, data_format, train=is_train)
        self.net_out = tf.identity(data, name='data')
        self.num_classes = num_classes

    def _resnet_block(self, filters, kernel, stride=(1,1), act_fn='relu', conv_1x1=0, name=None):
        """ """
        # data = self.net_out
        # shortcut  = data
        # bn_out    = self.batch_norm(data, act_fn, name=name+'_bn')
        # if conv_1x1:
        #     shortcut = self.convolution(bn_out, filters, (1,1), stride, pad='same', act_fn='',
        #                                 no_bias=True, name=name+'_1x1_conv')
        
        # net_out = self.convolution(bn_out, filters, kernel, stride, act_fn=act_fn, 
        #                             add_bn=True, name=name+'_conv1')
        # net_out = self.convolution(net_out, filters, kernel, (1,1), act_fn='', 
        #                             no_bias=True, name=name+'_conv2')

        # self.net_out = net_out + shortcut

        data = self.net_out
        shortcut  = data
        if conv_1x1:
            shortcut = self.convolution(data, filters, (1,1), stride, pad='same', act_fn='',
                                        add_bn=True, name=name+'_1x1_conv')
        
        net_out = self.convolution(data, filters, kernel, stride, act_fn=act_fn, 
                                    add_bn=True, name=name+'_conv1')
        net_out = self.convolution(net_out, filters, kernel, (1,1), act_fn='', 
                                    add_bn=True, name=name+'_conv2')

        net_out = net_out + shortcut
        self.net_out = tf.nn.relu(net_out, name=name+'_Relu')

    def _resnet_unit(self, num_blocks, filters, kernel, stride=1, act_fn='relu', name=None):
        """ """
        strides = (stride, stride)
        self._resnet_block(filters, kernel, strides, act_fn, conv_1x1=1, name=name+'_block0')
        for i in range(1, num_blocks):
            self._resnet_block(filters, kernel, (1,1), act_fn, name=name+'_block'+str(i))

    def __call__(self, num_stages=3, num_blocks=3, filters=[16, 32, 64], strides=[1,2,2]):
        """ """
        self.net_out = self.convolution(self.net_out, filters[0], (3,3), (1,1), 
                                        act_fn='relu', add_bn=True, name='Conv0')

        for k in range(num_stages):
            self._resnet_unit(num_blocks, filters[k], kernel=(3,3), 
                                stride=strides[k], name='stage'+str(k))

        net_out = self.pooling(self.net_out, 'avg', (8,8), name="global_pool")
        net_out = self.flatten(net_out)
        net_out = self.fully_connected(net_out, self.num_classes, name='FC_Softmax')
        return net_out

def snpx_net_create(num_classes, input_data, data_format="NHWC", is_training=True):
    """ """
    net = Resnet(input_data, data_format, num_classes, is_training)
    net_out = net(num_stages=3, num_blocks=3, filters=[16, 32, 64], strides=[1,2,2])
    return net_out
