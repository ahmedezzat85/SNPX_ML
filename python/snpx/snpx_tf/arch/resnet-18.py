import tensorflow as tf
from . tf_net import TFNet
from .. import tf_train_utils as tf_train

def _resnet_block_dw(net, num_filters, kernel, stride=(1,1), act='relu', conv_1x1=0, name=None):
    """ """
    shortcut  = net.net_out
    bn_out    = net.batch_norm(act=act, name=name+'_bn')
    if conv_1x1:
        shortcut = net.convolution(inputs=bn_out, num_filters=num_filters, kernel=(1,1), 
                                    stride=stride, pad='valid', no_bias=True, act_fn='',
                                    name=name+'_1x1_conv')
    
    net.conv_dw(inputs=bn_out, num_filters=num_filters, kernel=kernel, 
                        stride=stride, act_fn=act, add_bn=True, name=name+'_conv1')
    net.conv_dw(num_filters=num_filters, kernel=kernel, stride=(1,1), ptwise_conv=False,
                        act_fn=act, name=name+'_conv2')
    net.convolution(num_filters=num_filters, kernel=(1,1), stride=(1,1),
                        act_fn='', name=name+'_conv2_pt', no_bias=True)

    net.net_out = net.net_out + shortcut


def _resnet_block(net, num_filters, kernel, stride=(1,1), act='relu', conv_1x1=0, name=None):
    """ """
    shortcut  = net.net_out
    bn_out    = net.batch_norm(act_fn=act, name=name+'_bn')
    if conv_1x1:
        shortcut = net.convolution(inputs=bn_out, num_filters=num_filters, kernel=(1,1), 
                                    stride=stride, pad='valid', no_bias=True, act_fn='',
                                    name=name+'_1x1_conv')
    
    net.convolution(inputs=bn_out, num_filters=num_filters, kernel=kernel, 
                        stride=stride, act_fn=act, add_bn=True, name=name+'_conv1')
    net.convolution(num_filters=num_filters, kernel=kernel, stride=(1,1),
                        act_fn='', name=name+'_conv2')

    net.net_out = net.net_out + shortcut

def resnet_unit(net, num_blocks, num_filters, kernel, stride=(1,1), act='relu', dw_conv=False, name=None):
    """ """
    if dw_conv is False:
        _resnet_block(net, num_filters, kernel, stride, act, conv_1x1=1, name=name+'_block0')
        for i in (1, num_blocks):
            _resnet_block(net, num_filters, kernel, stride=(1,1), act=act, name=name+'_block'+str(i))
    else:        
        _resnet_block_dw(net, num_filters, kernel, stride, act, conv_1x1=1, name=name+'_block0')
        for i in (1, num_blocks):
            _resnet_block_dw(net, num_filters, kernel, stride=(1,1), act=act, name=name+'_block'+str(i))

def resnet_18(net, num_blocks=3, dw_conv=False):
    net.convolution(16, (3,3), stride=(1,1), name='Conv0', act_fn='')
    resnet_unit(net, num_blocks=num_blocks, num_filters=16, kernel=(3,3), stride=(1,1), 
                    act='relu', dw_conv=dw_conv, name='stage1')
    resnet_unit(net, num_blocks=num_blocks, num_filters=32, kernel=(3,3), stride=(2,2), 
                    act='relu', dw_conv=dw_conv, name='stage2')
    resnet_unit(net, num_blocks=num_blocks, num_filters=64, kernel=(3,3), stride=(2,2), 
                    act='relu', dw_conv=dw_conv, name='stage3')
    net.batch_norm(act_fn='relu', name='final_bn')
    net.pooling('avg', (8,8), name="global_pool")
    return net

def snpx_net_create(num_classes, 
                    input_data,
                    data_format="NHWC",
                    is_training=True):
    """ """
    dtype = input_data.dtype.base_dtype
    
    net = TFNet(input_data, data_format, train=is_training)
    net = resnet_18(net)
    net.flatten()
    net.fully_connected(num_classes, name='FC_Softmax')
    return net.net_out
