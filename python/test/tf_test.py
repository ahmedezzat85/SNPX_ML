import os
import numpy as np
import tensorflow as tf
from time import time
from datetime import datetime

def read_tf_record(tf_rec_file, shape=[32, 32, 3]):
    """ """
    idx = 0
    tf_reader = tf.python_io.tf_record_iterator(tf_rec_file)
    imgarr = tf.TensorArray(tf.uint8, dynamic_size=True, size=10000)
    labelarr = tf.TensorArray(tf.int64, dynamic_size=True, size=10000)
    for rec in tf_reader:
        rec_proto = tf.train.Example()
        rec_proto.ParseFromString(rec)
        feature = rec_proto.features.feature
        label = feature['label'].int64_list.value[0]
        image = tf.decode_raw(feature['image'].bytes_list.value[0], tf.uint8)
        image = tf.reshape(image, shape)
        imgarr = imgarr.write(idx, image)
        labelarr = labelarr.write(idx, label)
        idx += 1
    tf_reader.close()
    print (idx)
    return imgarr, labelarr

def main():
    """ """
    tf_rec = os.path.join(os.path.dirname(__file__), '..', 'snpx', 'datasets', 'CIFAR-10', 'CIFAR-10_val.tfrecords')
    sess = tf.Session()
    img, lab = read_tf_record(tf_rec)
    i,l = sess.run([img, lab])
    print (type(i), type(l))
    print (i.shape, l.shape)

if __name__ == '__main__':
    main()