import mxnet as mx

def Conv2d(data, num_filters, kernel, pad=(1,1), cudnn='fastest', act='relu'):
    Conv = mx.sym.Convolution(data=data, num_filter=num_filters, kernel=kernel, 
                             pad=pad, cudnn_tune=cudnn)
    Conv = mx.sym.Activation(data=Conv, act_type=act)
    return Conv
    
def snpx_net_create(num_classes, act='relu', use_fp16=False):
    """ """
    CNN = mx.sym.var('data')
    CNN = Conv2d(CNN, 64, (3,3))
    CNN = Conv2d(CNN, 64, (3,3))
    CNN = mx.sym.Pooling(data=CNN, pool_type='max', kernel=(2,2), stride=(2,2))

    CNN = Conv2d(CNN, 128, (3,3))
    CNN = Conv2d(CNN, 128, (3,3))
    CNN = mx.sym.Pooling(data=CNN, pool_type='max', kernel=(2,2), stride=(2,2))

    CNN = Conv2d(CNN, 256, (3,3))
    CNN = Conv2d(CNN, 256, (3,3))
    CNN = mx.sym.Pooling(data=CNN, pool_type='max', kernel=(2,2), stride=(2,2))

    CNN = mx.sym.Pooling(data=CNN, pool_type='avg', kernel=(4,4), stride=(4,4))
    CNN = mx.sym.Flatten(data=CNN)
    CNN = mx.sym.FullyConnected(data=CNN, num_hidden=num_classes)
    CNN = mx.sym.SoftmaxOutput(data=CNN, name='softmax')
    return CNN