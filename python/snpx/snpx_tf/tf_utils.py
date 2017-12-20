import os
import numpy as np
import tensorflow as tf
from time import time
from datetime import datetime

def read_tf_record(tf_rec_file, shape=[32, 32, 3]):
    """ """
    idx = 0
    tf_reader = tf.python_io.tf_record_iterator(tf_rec_file)
    for rec in tf_reader:
        rec_proto = tf.train.Example()
        rec_proto.ParseFromString(rec)
        feature = rec_proto.features.feature
        label = feature['label'].int64_list.value[0]
        image = tf.decode_raw(feature['image'].bytes_list.value[0], tf.uint8)
        image = tf.reshape(image, shape)
        idx += 1
    tf_reader.close()
    print (idx)