import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.data import TFRecordDataset, Iterator

_IMAGE_TFREC_STRUCTURE = {
        'image' : tf.FixedLenFeature([], tf.string),
        'label' : tf.FixedLenFeature([], tf.int64)
    }

def tf_create_data_iterator(
    batch_size, 
    train_set_file=None, 
    val_set_file=None, 
    shape=None, 
    dtype=tf.float32):
    """ """
    def tf_parse_record(tf_record):
        """ """
        feature = tf.parse_single_example(tf_record, features=_IMAGE_TFREC_STRUCTURE)
        image = tf.decode_raw(feature['image'], tf.uint8)
        image = tf.reshape(image, shape)
        image = tf.image.resize_images(image, [64, 64])
        label = tf.cast(feature['label'], tf.int64)
        image = tf.cast(image, dtype)
        return image, label

    if shape is None:
        raise ValueError("shape cannot be None")
    if val_set_file is None and train_set_file is None:
        raise ValueError("Both train_set_file and val_set_file are not specified")

    val_set       = None
    train_set     = None
    out_types     = None
    out_shapes    = None
    val_init_op   = None
    train_init_op = None

    # Create the validation dataset object
    if val_set_file is not None:
        val_set = TFRecordDataset(val_set_file)
        val_set = val_set.map(tf_parse_record)
        val_set = val_set.batch(batch_size)
        out_types = val_set.output_types
        out_shapes = val_set.output_shapes

    # Create the training dataset object
    if train_set_file is not None:
        train_set = TFRecordDataset(train_set_file)
        train_set = train_set.map(tf_parse_record)
        train_set = train_set.shuffle(buffer_size=batch_size * 1000)
        train_set = train_set.batch(batch_size)
        out_types = train_set.output_types
        out_shapes = train_set.output_shapes

    # Create a reinitializable iterator from both datasets
    iterator  = Iterator.from_structure(out_types, out_shapes)
    
    if train_set is not None:
        train_init_op   = iterator.make_initializer(train_set)
    
    if val_set is not None:
        val_init_op     = iterator.make_initializer(val_set)

    iter_op = iterator.get_next()
    return train_init_op, val_init_op, iter_op
