""" Load datasets in memory

"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import tensorflow as tf

try:
    import cPickle as pickle
except ImportError:
    import pickle

from .. backend import SNPX_DATASET_ROOT
from . tf_train_utils import tf_create_data_iterator

DATASETS = {'CIFAR-10': {'type'      : 'image_classification', 'num_classes': 10, 
                         'shape'     : (32,32,3), 
                         'train_file': 'CIFAR-10_train.tfrecords',
                         'val_file'  : 'CIFAR-10_val.tfrecords'}
        }


class TFDataset(object):
    """ 
    """
    def __init__(self, dataset_name, batch_size, for_training=True, dtype=tf.float32, data_format='NCHW'):
        if dataset_name not in DATASETS:
            raise ValueError('Dataset <%s> does not exist', dataset_name)
        
        with tf.device('/cpu:0'):
            dataset = DATASETS[dataset_name]
            dataset_dir = os.path.join(SNPX_DATASET_ROOT, dataset_name) 
            train_file  = os.path.join(dataset_dir, dataset['train_file']) if for_training else None
            val_file    = os.path.join(dataset_dir, dataset['val_file'])
            
            self.num_classes = dataset['num_classes']
            self.data_shape  = dataset['shape']
            self.train_set_init_op, self.eval_set_init_op, self.iter_op = \
                tf_create_data_iterator(batch_size, train_file, val_file, self.data_shape, dtype)
            
            if dataset['type'] == 'image_classification':
                self.images, labels = self.iter_op
                self.labels  = tf.one_hot(labels, self.num_classes)
                if data_format.startswith('NC'):
                    self.images = tf.transpose(self.images, [0, 3, 1, 2])

##########################################################
def _int64_feature(value):
    if isinstance(value, list):
        val = value
    else:
        val = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=val))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def images_to_tfrecord(tf_rec_file, images, labels):
    """ """
    # Create TF Record Writer 
    tf_rec_writer = tf.python_io.TFRecordWriter(tf_rec_file)

    # Write the images and labels to TFRecord file
    print ("data_shape = ", images.shape)
    N = images.shape[0]
    
    for i in range(N):
        # Format the record
        image_raw = images[i].tostring()
        features = tf.train.Features(feature={
            'image' : _bytes_feature(image_raw),
            'label' : _int64_feature(labels[i])
        })
        # Write the record to the file
        tf_rec_proto = tf.train.Example(features=features)
        tf_rec_writer.write(tf_rec_proto.SerializeToString())

    tf_rec_writer.close()

#----------------
# CIFAR-10 Loader   
#----------------
class CIFAR10(object):
    """
    """
    def __init__(self):
        self.data_dir    = os.path.join(SNPX_DATASET_ROOT, "CIFAR-10")
        self.train_file  = os.path.join(self.data_dir, "CIFAR-10_train.tfrecords")
        self.val_file    = os.path.join(self.data_dir, "CIFAR-10_val.tfrecords")
        
    def _load_CIFAR_batch(self, batch_file):
        """ Read a CIFAR-10 batch file into numpy arrays """
        with open(os.path.join(self.data_dir, batch_file), 'rb') as f:
            if sys.version_info.major == 3:
                datadict = pickle.load(f, encoding='bytes')
            else:
                datadict = pickle.load(f)
            x = datadict[b'data']
            y = datadict[b'labels']
            
            y = np.array(y)
            x = np.reshape(x,[-1, 3, 32, 32])
            x = x.swapaxes(1,3)
            return x, y

    def write_to_tfrecord(self):
        """ """
        X_Train, Y_Train, X_Val, Y_Val = self.get_raw_data()
        images_to_tfrecord(self.train_file, X_Train, Y_Train)
        images_to_tfrecord(self.val_file, X_Val, Y_Val)

    def get_raw_data(self):
        """   """
        for b in range(1,6):
            f = 'data_batch_' + str(b)
            xb, yb = self._load_CIFAR_batch(f)
            if b > 1:
                x_train = np.concatenate((x_train, xb))
                y_train = np.concatenate((y_train, yb))
                del xb, yb
            else:
                x_train = xb
                y_train = yb

        x_val, y_val    = self._load_CIFAR_batch('test_batch')
        return x_train, y_train, x_val, y_val
