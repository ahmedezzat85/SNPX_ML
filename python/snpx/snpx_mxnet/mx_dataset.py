""" Load datasets in memory

"""
from __future__ import absolute_import

import os
import cv2
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import threading

try:
    import cPickle as pickle
except ImportError:
    import pickle

from . mx_train_utils import mx_create_data_iterator
from .. backend import SNPX_DATASET_ROOT
from .. util import snpx_create_dir


#----------------
# CONSTANTS   
#----------------
MNIST_ROOT      = os.path.join(SNPX_DATASET_ROOT, "MNIST")
CIFAR10_ROOT    = os.path.join(SNPX_DATASET_ROOT, "CIFAR-10")
UCF101_ROOT     = os.path.join(SNPX_DATASET_ROOT, "UCF-101")


DATASETS = {'CIFAR-10': {'type'      : 'image_classification', 'num_classes': 10, 
                         'shape'     : (3, 32, 32), 
                         'train_file': 'CIFAR-10_train.mxrec',
                         'val_file'  : 'CIFAR-10_val.mxrec'}
        }


class MxDataset(object):
    """ 
    """
    def __init__(self, dataset_name, batch_size, for_training=True, dtype=np.float32):
        if dataset_name not in DATASETS:
            raise ValueError('Dataset <%s> does not exist', dataset_name)
        
        dataset = DATASETS[dataset_name]
        dataset_dir = os.path.join(SNPX_DATASET_ROOT, dataset_name) 
        train_file  = os.path.join(dataset_dir, dataset['train_file']) if for_training else None
        val_file    = os.path.join(dataset_dir, dataset['val_file'])
            
        self.num_classes = dataset['num_classes']
        self.data_shape  = dataset['shape']
        self.mx_train_iter, self.mx_eval_iter = \
            mx_create_data_iterator(batch_size, train_file, val_file, self.data_shape, dtype)            

##########################################################

#----------------
# CIFAR-10 Loader   
#----------------
class CIFAR10(object):
    """
    """
    def __init__(self, batch_size=128, data_aug=False):
        self.n_class    = 10
        self.batch_size = batch_size
        self.data_aug   = data_aug
        self.data_shape = (3, 32, 32)
        self.data_dir   = CIFAR10_ROOT

    def load_train_data(self, split=False):
        """
        """
        train_aug_types = None
        test_aug_types  = None
        if self.data_aug == True:
            train_aug_types = {"horiz_flip": 2, "color_jitter": 1}
            test_aug_types  = {"horiz_flip": 2, "color_jitter": 1}
            
        self.train_iter = SPXImageIter(
            batch_size          = self.batch_size,
            data_shape          = self.data_shape,
            path_imgrec         = os.path.join(self.data_dir, "train.rec"),
            path_imgidx         = os.path.join(self.data_dir, "train.lst"),
            shuffle             = False,
            aug_types           = train_aug_types
        )
        self.val_iter = SPXImageIter(
            batch_size          = self.batch_size,
            data_shape          = self.data_shape,
            path_imgrec         = os.path.join(self.data_dir, "val.rec"),
            aug_types           = test_aug_types
        )
        return self.train_iter, self.val_iter

    def load_test_data(self):
        """
        """
        self.val_iter = SPXImageIter(
            batch_size          = self.batch_size,
            data_shape          = self.data_shape,
            path_imgrec         = os.path.join(self.data_dir, "val.rec")
        )
        return self.val_iter


# #----------------------------------------------------#
# # UCF-101 Loader   
# #----------------------------------------------------#

# def vclip_2_ndarray(in_file, label, out_file):
#     """ Convert a video file to mxnet.ndarray array
#     """
#     vframe_list = []
#     cap = cv2.VideoCapture(in_file)
#     while(cap.isOpened()):
#         ret, vframe = cap.read()
#         if not ret:
#             break
#         vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
#         vframe = cv2.resize(vframe, (171, 128))
#         vframe = nd.array(vframe)
#         vframe,_ = mx.img.center_crop(vframe, (112, 112))
#         vframe = nd.swapaxes(vframe, 0, 2)
#         vframe = nd.expand_dims(vframe, axis=1)
#         vframe_list.append(vframe)
    
#     if len(vframe_list) is not 0:
#         v_arr = nd.concat(*vframe_list, dim=1)
#         label_arr = nd.empty((1,1))
#         label_arr[0] = label
#         data = {"data": v_arr.astype(np.uint8), "label": label_arr}
#         nd.save(out_file, data)
#     else:
#         raise ValueError("File %s Open Failed", in_file)

# class PreprocessThread(threading.Thread):
#     """
#     """
#     def __init__(self, in_file, label, out_file):
#         threading.Thread.__init__(self)
#         self.in_file    = in_file
#         self.label      = label
#         self.out_file   = out_file

#     def run(self):
#         vclip_2_ndarray(self.in_file, self.label, self.out_file)
        
# class VideoClipPreprocessor(object):
#     """
#     """
#     def __init__(self, max_threads=1):
#         self.max_threads = max_threads
#         if max_threads > 1:
#             self.max_threads += 1

#     def process_clip(self, in_file, label, out_file):
#         """
#         """
#         if self.max_threads > 1:
#             while (threading.active_count() >= self.max_threads):
#                 pass
#             th = PreprocessThread(in_file, label, out_file)
#             th.start()
#         else:
#             vclip_2_ndarray(in_file, label, out_file)

# class UCF101(object):
#     """ UCF-101 dataset loader.

#     Parameters
#     ----------
#     batch_size : int
#         Training mini-batch size.
#     data_aug : bool
#     train_split : int
#         ID of the train_split. Allowed values are 0,1,2 and 3.
#         0 means all training data are included, while 1,2 or 3 is the ID of
#         the partition of the UCF-101.
#     """
#     def __init__(self, batch_size=128, data_aug=False, train_split=1):
#         if train_split not in range(4):
#             raise ValueError("Invalid train_split %d. Allowed values are [0:3]", train_split)
        
#         self.n_class        = 101
#         self.batch_size     = batch_size
#         self.data_aug       = data_aug
#         self.data_shape     = (3, 16, 112, 112)
#         self.raw_data_dir   = os.path.join(UCF101_ROOT, "data")
#         self.train_data_dir = os.path.join(UCF101_ROOT, "train")
#         self.test_data_dir  = os.path.join(UCF101_ROOT, "test")
#         self.class_names = []
#         if train_split == 0:
#             self.split_list = [1,2,3]
#         else:
#             self.split_list = [train_split]

#     def preprocess(self, preprocess_threads=1, context="GPU"):
#         """ Preprocess UCF-101 dataset for fast training.
        
#         The training/testing video clips of the UCF-101 dataset is defined as a
#         list in a text file. Each line of the text file contains a video clip 
#         path and the corresponding label with a space delimiter. the exact format is:
#         "root/video_clip_name label"
        
#         The preprocessing is to read the video clips and convert them to mxnet.ndarray
#         arrays. Then save the ndarray to a file in a dict format 
#         {"data": data_array, "label": label}
#         """
#         # Get class names
#         class_idx = {}
#         with open(os.path.join(UCF101_ROOT, "ucfTrainTestlist", "classInd.txt"), "r") as fin:
#             for line in fin:
#                 label_idx, class_name = line.strip().split(' ')
#                 class_idx.update({class_name: label_idx})

#         video_preproc = VideoClipPreprocessor(max_threads=preprocess_threads)

#         if context == "GPU":
#             ctx = mx.gpu()
#         else:
#             ctx = mx.cpu()

#         with mx.Context(ctx):
#             # Training data
#             prfx = "train"
#             save_dir = self.train_data_dir
#             snpx_create_dir(save_dir) ## Create a directory for saving train/test preprocessed data
#             fout = open(os.path.join(save_dir, prfx+".list"), "w")
#             for i in self.split_list:
#                 list_file = os.path.join(UCF101_ROOT, "ucfTrainTestlist", prfx+"list0"+str(i)+".txt")
#                 with open(list_file, "r") as fin:
#                     for line in fin:
#                         vfpath, label_str = line.strip().split(' ')
#                         root, vfile   = vfpath.strip().split('/')
#                         data_fname,_ = vfile.split('.')  # Strip the file name extension
#                         wline = data_fname+'\t'+label_str
#                         print (wline)
#                         fout.write(wline+'\n')
#                         video_preproc.process_clip(in_file=os.path.join(self.raw_data_dir, vfpath),
#                                                     out_file=os.path.join(save_dir, data_fname),
#                                                     label=int(label_str))
#             fout.close()      

#             # Test data
#             prfx = "test"
#             save_dir = self.test_data_dir
#             snpx_create_dir(save_dir) ## Create a directory for saving test preprocessed data
#             fout = open(os.path.join(save_dir, prfx+".list"), "w")
#             for i in self.split_list:
#                 list_file = os.path.join(UCF101_ROOT, "ucfTrainTestlist", prfx+"list0"+str(i)+".txt")
#                 with open(list_file, "r") as fin:
#                     for line in fin:
#                         vfpath = line.strip()
#                         root, vfile   = vfpath.strip().split('/')
#                         data_fname,_ = vfile.split('.')  # Strip the file name extension
#                         label = class_idx[root]
#                         wline = data_fname+'\t'+label
#                         print (wline)
#                         fout.write(wline+'\n')
#                         video_preproc.process_clip(in_file=os.path.join(self.raw_data_dir, vfpath),
#                                                     out_file=os.path.join(save_dir, data_fname),
#                                                     label=int(label_str))
#             fout.close()      
