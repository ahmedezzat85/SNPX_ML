import os
import numpy as np
import tensorflow as tf
from time import time
from datetime import datetime
from tensorflow.contrib.data import TFRecordDataset, Iterator
from .tf_train_utils import tf_parse_record as get_record
from .tf_dataset import tf_get_dataset_object
from ..backend import SNPX_DATASET_ROOT

def read_tf_record(dataset):
    """ """
    idx = 0
    tf_rec_file = os.path.join(SNPX_DATASET_ROOT, dataset, dataset+"_train.tfrecords")
    tf_reader = tf.python_io.tf_record_iterator(tf_rec_file)
    for rec in tf_reader:
        rec_proto = tf.train.Example()
        rec_proto.ParseFromString(rec)
        feature = rec_proto.features.feature
        label = feature['label'].int64_list.value[0]
        image = tf.decode_raw(feature['image_raw'].bytes_list.value[0], tf.uint8)
        image = tf.reshape(image, [H, W, C])
        idx += 1
    tf_reader.close()
    print (idx)

def data_test(dataset_name, batch_size):
    """ """
    tf.logging.set_verbosity(tf.logging.INFO)
    tf_rec_file = os.path.join(SNPX_DATASET_ROOT, dataset_name, dataset_name+"_train.tfrecords")
    dataset = TFRecordDataset(tf_rec_file)
    dataset = dataset.map(get_record)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    train_iter = dataset.make_initializable_iterator()
    train_iter_next = train_iter.get_next()
    start_time  = datetime.now()
    tf.logging.info("Started at : " + start_time.strftime("%Y-%m-%d %H:%M:%S"))
    with tf.Session() as sess:
        sess.run(train_iter.initializer)
        i = 0
        while True:
            try:
                im,l = sess.run(train_iter_next)
                print (i)
                i += 1
            except tf.errors.OutOfRangeError:
                break

    tf.logging.info("Finished at   : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    tf.logging.info("Elapsed Time  : " + str(datetime.now() - start_time))
