from __future__ import absolute_import

import os
import cv2
import random
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd


def mx_create_data_iterator(batch_size, train_set_file=None, val_set_file=None, 
                            shape=None, dtype=np.float32):
        train_iter = None
        val_iter   = None
        if train_set_file is not None:
            train_iter = mx.img.ImageIter(batch_size, shape, path_imgrec=train_set_file)
        if val_set_file is not None:
            val_iter = mx.img.ImageIter(batch_size, shape, path_imgrec=val_set_file)
            # MxImageIter(batch_size=batch_size, data_shape=shape, path_imgrec=val_set_file)
        return train_iter, val_iter

#----------------------------------------------------#
# IMAGE DATA ITERATOR   
#----------------------------------------------------#
class MxImageIter(mx.io.DataIter):
    """Image data iterator with a large number of augumentation choices.
    Supports reading from both .rec files and raw image files with image list.

    To load from .rec files, please specify path_imgrec. Also specify path_imgidx
    to use data partition (for distributed training) or shuffling.

    To load from raw image files, specify path_imglist and path_root.

    Parameters
    ----------
    batch_size : int
        Number of examples per batch
    data_shape : tuple
        Data shape in (channels, height, width).
        For now, only RGB image with 3 channels is supported.
    label_width : int
        dimension of label
    path_imgrec : str
        path to image record file (.rec).
        Created with tools/im2rec.py or bin/im2rec
    path_imgidx : str
        Path to image index file. Needed for partition and shuffling when using .rec source.
    shuffle : bool
        Whether to shuffle all images at the start of each iteration.
        Can be slow for HDD.
    aug_flag : bool
        Whether to use more augmentation options to extend the dataset.
    """
    def __init__(self, batch_size, data_shape, label_width=1, path_imgrec=None, 
                    path_imgidx=None, shuffle=False, rgb_mean=True, aug_types=None):
        # Validate the data shape
        assert len(data_shape) == 3 and data_shape[0] == 3
        super().__init__()

        self.seq            = None
        self.shuffle        = shuffle
        self.batch_size     = batch_size
        self.data_shape     = data_shape
        self.label_width    = label_width
        self.auglist        = []
        self.num_data       = 0
        if(rgb_mean == True):
            self.rgb_mean   = nd.array([125.345, 122.942, 113.839])
        else:
            self.rgb_mean   = nd.array([0, 0, 0])
        
        # Init record io iterator
        if path_imgrec:
            if path_imgidx:
                self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
                if shuffle:
                    self.seq = list(self.imgrec.keys)
            else:
                self.imgrec = mx.recordio.MXRecordIO(path_imgrec, 'r')
        else:
            raise ValueError("No image source is given")
        
        # Set data and label shapes
        self.provide_data = [('data', (batch_size,) + data_shape)]
        if label_width > 1:
            self.provide_label = [('softmax_label', (batch_size, label_width))]
        else:
            self.provide_label = [('softmax_label', (batch_size,))]

        # Create Augmenters List
        self.auglist.append(mx.img.CreateAugmenter(data_shape))   # Default one is the center_crop augmenter
        if aug_types is not None:
            for aug_name, aug_repeat in aug_types.items():
                if aug_name == "horiz_flip" or aug_name == "rand_mirror":
                    augm = [mx.img.CreateAugmenter(data_shape, rand_mirror=True) for i in range(aug_repeat)]
                elif aug_name == "color_jitter":
                    augm = [mx.img.CreateAugmenter(data_shape, brightness=np.random.rand() * 50, contrast=np.random.rand() * 50) for i in range(aug_repeat)]
                self.auglist.extend(augm)

        self.cur = 0
        self.reset()

    def reset(self):
        if self.shuffle:
            random.shuffle(self.seq)
        self.imgrec.reset()
        self.cur = 0

    def next_sample(self):
        """helper function for reading in next sample"""
        if self.seq is not None:
            if self.cur >= len(self.seq):
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            s = self.imgrec.read_idx(idx)
            header, img = mx.recordio.unpack(s)
            return header.label, img
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = mx.recordio.unpack(s)
            return header.label, img

    def next(self):
        """ Get next batch from the iterator.
        """
        batch_size  = self.batch_size
        c, h, w     = self.data_shape
        batch_data  = nd.empty((batch_size, c, h, w))
        batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                # Get Image from recordio
                label, s = self.next_sample()
                data = [mx.img.imdecode(s)]
                if len(data[0].shape) == 0:
                    logging.debug('Invalid image, skipping.')
                    continue
                
                # Augmentation
                #   Create multiple versions of the image according to the list 
                #   of defined augmenters. Append eash augmenter result to the 
                #   data list. This will extend the dataset.
                data_list = []
                for augmenter in self.auglist:
                    for aug in augmenter:
                        data = [ret for src in data for ret in aug(src)]
                    for src in data:
                        src -= self.rgb_mean
                        data_list.append(src)
                
                for d in data_list:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    batch_data[i][:] = nd.transpose(d, axes=(2, 0, 1))
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if not i:
                raise StopIteration
        self.num_data += self.batch_size
        return mx.io.DataBatch([batch_data], [batch_label], batch_size-1-i)
