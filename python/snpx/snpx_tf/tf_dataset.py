""" Load datasets in memory

"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.data import TFRecordDataset, Iterator

from .. backend import SNPX_DATASET_ROOT
from .. util import DictToAttrs

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
        
        print ('Data Aug ', data_aug)
        dataset = DATASETS[dataset_name]
        dataset_dir = os.path.join(SNPX_DATASET_ROOT, dataset_name) 
        train_file  = os.path.join(dataset_dir, dataset['train_file'])
        val_file    = os.path.join(dataset_dir, dataset['val_file'])
        test_file   = os.path.join(dataset_dir, dataset['test_file'])
        mean_image  = os.path.join(dataset_dir, dataset['mean_img'])

        self.dtype       = dtype
        self.num_classes = dataset['num_classes']
        self.shape       = dataset['shape']
        self.mean_img    = np.fromfile(mean_image).reshape(self.shape)
        # Generate Dataset Iterators 
        if for_training is True:
            train_preproc = self._train_preproc if data_aug is True else self._test_preproc
            data = [
                {'file': train_file , 'preproc': train_preproc      , 'shuffle': True},
                {'file': val_file   , 'preproc': self._test_preproc , 'shuffle': False}
            ]
            iter_op, init_ops = self._create_data_iterators(data, batch_size, dtype)
            self.train_set_init_op, self.eval_set_init_op = init_ops
        else:
            data = [
                {'file': test_file  , 'preproc': self._test_preproc , 'shuffle': False}
            ]
            iter_op, init_ops = self._create_data_iterators(data, batch_size, dtype)
            self.eval_set_init_op = init_ops
            
        if dataset['type'] == 'image_classification':
            self.images, labels = iter_op
            self.labels  = tf.one_hot(labels, self.num_classes)
            if data_format.startswith('NC'):
                self.images = tf.transpose(self.images, [0, 3, 1, 2])

    def _train_preproc(self, image):
        """ """
        im_out = tf.image.resize_image_with_crop_or_pad(image, self.shape[0] + 8, self.shape[1] + 8)
        im_out = tf.random_crop(im_out, self.shape)
        im_out = tf.image.random_flip_left_right(im_out)
        im_out = tf.cast(im_out, self.dtype)
        return im_out

    def _test_preproc(self, image):
        """ """
        image = tf.cast(image, self.dtype)
        return image

    def _create_data_iterators(self, tf_rec_list, batch_size, dtype=tf.float32):
        """ """
        def tf_parse_record(tf_record):
            """ """
            feature = tf.parse_single_example(tf_record, features=_IMAGE_TFREC_STRUCTURE)
            image = tf.decode_raw(feature['image'], tf.uint8)
            image = tf.reshape(image, self.shape)
            label = tf.cast(feature['label'], tf.int64)
            return image, label

        data_iter = None
        iter_list = []
        for rec in tf_rec_list:
            rec = DictToAttrs(rec)
            dataset = TFRecordDataset(rec.file)
            dataset = dataset.map(tf_parse_record)
            dataset = dataset.map(lambda image, label: (rec.preproc(image), label))#, batch_size)
            if rec.shuffle: dataset = dataset.shuffle(buffer_size=50000)
            dataset = dataset.batch(batch_size)
            out_types = dataset.output_types
            out_shapes = dataset.output_shapes

            if data_iter is None:
                # Create a reinitializable iterator
                data_iter  = Iterator.from_structure(out_types, out_shapes)
            iter_init = data_iter.make_initializer(dataset)
            iter_list.append(iter_init)

        return data_iter.get_next(), iter_list

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
        self.mean_image  = os.path.join(self.data_dir, cifar10_dict['mean_img'])
        
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
        np.mean(X_Train, axis=0).tofile(self.mean_image)
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
