""" Load datasets in memory

"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.data import TFRecordDataset, Iterator

from .. backend import SNPX_DATASET_ROOT

try:
    import cPickle as pickle
except ImportError:
    import pickle

_IMAGE_TFREC_STRUCTURE = {
        'image' : tf.FixedLenFeature([], tf.string),
        'label' : tf.FixedLenFeature([], tf.int64)
    }

DATASETS = {'CIFAR-10': {'type'      : 'image_classification', 'num_classes': 10, 
                         'shape'     : (32,32,3), 
                         'train_file': 'CIFAR-10_train.tfrecords',
                         'val_file'  : 'CIFAR-10_val.tfrecords',
                         'test_file' : 'CIFAR-10_test.tfrecords',
                         'mean_img'  : 'CIFAR-10.mean'}
        }


class TFDataset(object):
    """ 
    """
    def __init__(self,
                 dataset_name,
                 batch_size,
                 for_training=True,
                 dtype=tf.float32,
                 data_format='NCHW',
                 data_aug=False):
        # Search for the dataset
        if dataset_name not in DATASETS:
            raise ValueError('Dataset <%s> does not exist', dataset_name)
        
        # Process on CPU
        with tf.device('/cpu:0'):
            dataset = DATASETS[dataset_name]
            dataset_dir = os.path.join(SNPX_DATASET_ROOT, dataset_name) 
            self.train_file  = os.path.join(dataset_dir, dataset['train_file']) if for_training else None
            self.val_file    = os.path.join(dataset_dir, dataset['val_file'])
            
            self.num_classes = dataset['num_classes']
            self.shape  = dataset['shape']
            self.mean_img    = np.fromfile(os.path.join(dataset_dir, dataset['mean_img'])).reshape(self.shape)
            self.tf_create_data_iterator(batch_size, dtype, data_aug)
            
            if dataset['type'] == 'image_classification':
                self.images, labels = self.iter_op
                self.labels  = tf.one_hot(labels, self.num_classes)
                if data_format.startswith('NC'):
                    self.images = tf.transpose(self.images, [0, 3, 1, 2])

    def preprocess(self, image):
        """ """
        # im_out = tf.subtract(image, self.mean_img)
        # im_out = image / 255
        im_out = tf.image.resize_image_with_crop_or_pad(image, self.shape[0] + 8, self.shape[1] + 8)
        im_out = tf.random_crop(im_out, self.shape)
        im_out = tf.image.random_flip_left_right(im_out)
        im_out = tf.image.per_image_standardization(im_out)
        return im_out

    def tf_create_data_iterator(self,
                                batch_size, 
                                dtype=tf.float32,
                                data_aug=False):
        """ """
        def preprocess(image):
            return self.preprocess(image)

        def tf_parse_record(tf_record):
            """ """
            feature = tf.parse_single_example(tf_record, features=_IMAGE_TFREC_STRUCTURE)
            image = tf.decode_raw(feature['image'], tf.uint8)
            image = tf.reshape(image, self.shape)
            label = tf.cast(feature['label'], tf.int64)
            image = tf.cast(image, dtype)
            return image, label

        val_set       = None
        train_set     = None
        out_types     = None
        out_shapes    = None
        val_init_op   = None
        train_init_op = None

        # Create the validation dataset object
        if self.val_file is not None:
            val_set = TFRecordDataset(self.val_file)
            val_set = val_set.map(tf_parse_record)
            val_set = val_set.batch(batch_size)
            out_types = val_set.output_types
            out_shapes = val_set.output_shapes

        # Create the training dataset object
        if self.train_file is not None:
            train_set = TFRecordDataset(self.train_file)
            train_set = train_set.map(tf_parse_record)
            train_set = train_set.map(lambda image, label: (preprocess(image), label))
            train_set = train_set.shuffle(buffer_size=50000) # TODO Remove the hardcoded value
            train_set = train_set.batch(batch_size)
            out_types = train_set.output_types
            out_shapes = train_set.output_shapes

        # Create a reinitializable iterator from both datasets
        iterator  = Iterator.from_structure(out_types, out_shapes)
        
        if train_set is not None:
            self.train_set_init_op = iterator.make_initializer(train_set)
        
        if val_set is not None:
            self.eval_set_init_op = iterator.make_initializer(val_set)

        self.iter_op = iterator.get_next()
        return self.train_set_init_op, self.eval_set_init_op, self.iter_op

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
        cifar10_dict = DATASETS['CIFAR-10']
        self.data_dir    = os.path.join(SNPX_DATASET_ROOT, "CIFAR-10")
        self.train_file  = os.path.join(self.data_dir, cifar10_dict['train_file'])
        self.val_file    = os.path.join(self.data_dir, cifar10_dict['val_file'])
        self.test_file   = os.path.join(self.data_dir, cifar10_dict['test_file'])
        
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
        X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test = self.get_raw_data()
        images_to_tfrecord(self.train_file, X_Train, Y_Train)
        images_to_tfrecord(self.val_file, X_Val, Y_Val)
        images_to_tfrecord(self.test_file, X_Test, Y_Test)

    def get_raw_data(self, val_split=0.1):
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

        x_test, y_test = self._load_CIFAR_batch('test_batch')

        # Perform Train/Val Split
        if val_split:
            num_train = len(x_train)
            split_idx = int(num_train - num_train * val_split)
            x_tr, x_val = np.split(x_train, [split_idx])
            y_tr, y_val = np.split(y_train, [split_idx])
        else:
            x_tr = x_train
            y_tr = y_train
            x_val = x_test
            y_val = y_test

        return x_tr, y_tr, x_val, y_val, x_test, y_test
