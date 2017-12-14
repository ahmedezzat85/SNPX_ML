""" cnn-1
"""
from __future__ import absolute_import

import numpy as np
from .net import SNPXNet
import mxnet as mx

class CNN(SNPXNet):
    """
    """
    def __init__(self, fp16=False, batch_norm=False):
        super(CNN, self).__init__(fp16, batch_norm)

    def Conv_Blk11(self, data, num_filter):
        Conv = self.Convolution(data=data, num_filter=num_filter, kernel=(1,1), pad=(0,0), incr=False)
        Conv = self.Convolution(data=Conv, num_filter=int(num_filter*0.5), kernel=(1,3), pad=(0,1), incr=False)
        Conv = self.Convolution(data=Conv, num_filter=int(num_filter*0.5), kernel=(3,1), pad=(1,0), incr=False)
        return Conv

    def Conv_Blk12(self, data, num_filter):
        Conv = self.Convolution(data=data, num_filter=num_filter, kernel=(1,1), pad=(0,0), incr=False)
        Conv = self.Convolution(data=Conv, num_filter=int(num_filter*0.5), kernel=(1,3), pad=(0,1), incr=False)
        Conv = self.Convolution(data=Conv, num_filter=int(num_filter*0.5), kernel=(3,1), pad=(1,0), incr=False)
        Conv = self.Convolution(data=Conv, num_filter=int(num_filter*0.5), kernel=(1,3), pad=(0,1), incr=False)
        Conv = self.Convolution(data=Conv, num_filter=int(num_filter*0.5), kernel=(3,1), pad=(1,0), incr=False)
        return Conv

    def Conv_Block1(self, name, num_filter):
        Conv1 = self.Conv_Blk11(self.Net, num_filter/2)
        Conv2 = self.Conv_Blk12(self.Net, num_filter/2)
        self.Concat([Conv1, Conv2])

    def Conv_Blk21(self, data, num_filter):
        Conv = self.Convolution(data=data, num_filter=num_filter, kernel=(1,1), pad=(0,0), incr=False)
        Conv = self.Convolution(data=Conv, num_filter=int(num_filter*0.5), kernel=(1,3), pad=(0,1), incr=False)
        Conv = self.Convolution(data=Conv, num_filter=int(num_filter*0.5), kernel=(3,1), pad=(1,0), incr=False)
        return Conv

    def Conv_Blk22(self, data, num_filter):
        Conv = self.Convolution(data=data, num_filter=num_filter, kernel=(1,1), pad=(0,0), incr=False)
        Conv = self.Convolution(data=Conv, num_filter=int(num_filter*0.5), kernel=(1,3), pad=(0,1), incr=False)
        Conv = self.Convolution(data=Conv, num_filter=int(num_filter*0.5), kernel=(3,1), pad=(1,0), incr=False)
        Conv = self.Convolution(data=Conv, num_filter=int(num_filter*0.5), kernel=(1,3), pad=(0,1), incr=False)
        Conv = self.Convolution(data=Conv, num_filter=int(num_filter*0.5), kernel=(3,1), pad=(1,0), incr=False)
        return Conv

    def Conv_Block2(self, name, num_filter):
        Conv1 = self.Conv_Blk21(self.Net, num_filter/4)
        Conv2 = self.Conv_Blk21(self.Net, num_filter/4)
        Conv3 = self.Conv_Blk21(self.Net, num_filter/4)
        Conv4 = self.Conv_Blk21(self.Net, num_filter/4)
        self.Concat([Conv1, Conv2, Conv3, Conv4])


def snpx_net_create(num_classes, fp16=False, batch_norm=True):
    """
    """
    snpx_cnn = CNN(fp16=fp16, batch_norm=batch_norm)

    # CONV1 3x3/1,64
    snpx_cnn.Convolution(name='Conv1a', num_filter=64, kernel=(3,3), pad=(1,1))
    snpx_cnn.MaxPool(name='Pool1', kernel=(2,2), stride=(2,2))

    # CONV2 3x3/1,128 
    snpx_cnn.Conv_Block1(name="ConvBlk1", num_filter=128)
    snpx_cnn.MaxPool(name='Pool2', kernel=(2,2), stride=(2,2))

    # CONV3 3x3/1,256 
    snpx_cnn.Conv_Block1(name="ConvBlk2", num_filter=256)
    snpx_cnn.Conv_Block1(name="ConvBlk3", num_filter=256)
    snpx_cnn.MaxPool(name='Pool3', kernel=(2,2), stride=(2,2))

    # CONV4 3x3/1,512 
    snpx_cnn.Conv_Block2(name="ConvBlk4", num_filter=512)
    snpx_cnn.Conv_Block2(name="ConvBlk5", num_filter=512)
    snpx_cnn.Conv_Block2(name="ConvBlk6", num_filter=512)
    snpx_cnn.Conv_Block2(name="ConvBlk7", num_filter=512)
    snpx_cnn.AvgPool(name='Pool4', kernel=(4,4), stride=(4,4))

    # Fully Connected
    snpx_cnn.Flatten()

    # Softmax Classifier
    snpx_cnn.Softmax(num_classes=num_classes)
    return snpx_cnn.Net