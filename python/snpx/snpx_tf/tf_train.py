import os
import numpy as np
from time import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.data import TFRecordDataset, Iterator
from .arch.mlp import snpx_net_create
from .. backend import SNPX_DATASET_ROOT


def tf_parse_record(tf_record):
    """ """
    feature = tf.parse_single_example(tf_record, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    image = tf.decode_raw(feature['image'], tf.uint8)
    label = tf.cast(feature['label'], tf.int32)
    image = tf.reshape(image, [32, 32, 3])
    image = tf.cast(image, tf.float32)
    return image, label

def tf_create_iterator(dataset, batch_size):
    """ """
    dataset_prefix = os.path.join(SNPX_DATASET_ROOT, dataset, dataset)
    train_rec_file = dataset_prefix + "_train.tfrecords"
    val_rec_file   = dataset_prefix + "_val.tfrecords"

    # Create the training dataset object
    train_set = TFRecordDataset(train_rec_file)
    train_set = train_set.map(tf_parse_record, num_threads=4, output_buffer_size=1000)
    train_set = train_set.shuffle(buffer_size=50000)
    train_set = train_set.batch(batch_size)

    # Create the validation dataset object
    val_set = TFRecordDataset(val_rec_file)
    val_set = val_set.map(tf_parse_record)
    val_set = val_set.batch(batch_size)

    # Create a reinitializable iterator from both datasets
    iterator  = Iterator.from_structure(train_set.output_types, train_set.output_shapes)
    train_init_op   = iterator.make_initializer(train_set)
    val_init_op     = iterator.make_initializer(val_set)
    iter_op = iterator.get_next()
    return train_init_op, val_init_op, iter_op

def tf_create_train_op(images, labels):
    """ """
    # Forward Propagation
    onehot_labels  = tf.one_hot(labels, 10)
    predictions    = snpx_net_create(10, images)

    opt = tf.train.AdamOptimizer()
    
    # Compute the loss and the train_op
    global_step = tf.train.get_or_create_global_step()
    # update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    loss = tf.losses.softmax_cross_entropy(onehot_labels, predictions)
    total_loss = tf.losses.get_total_loss()
    t = opt.minimize(total_loss, global_step)
    train_op = [t, total_loss]
    result  = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(onehot_labels, axis=1))
    val_op  = tf.reduce_mean(tf.cast(result, tf.float32))

    summ = tf.summary.scalar("loss", total_loss)

    return train_op, val_op, global_step, summ

def tf_train_loop(tf_sess, train_op, global_step, tb_writer, summ, batch_size):
    """ """
    count = 0
    epoch_start_time = time()
    last_log_tick    = epoch_start_time
    last_log_batch   = tf_sess.run(global_step)
    while True:
        try:
            loss, s, i = tf_sess.run([train_op, summ, global_step])
            tb_writer.add_summary(s, i)
            tb_writer.flush()
            count += 1
            print (count, i)
            if (i - last_log_batch) >= 10:
                elapsed = time() - last_log_tick
                freq = ((i - last_log_batch)  * batch_size/ elapsed)
                last_log_batch = i
                last_log_tick  = time()
                tf.logging.info('Batch[%d]\tloss: %.3f\tspeed: %.3f samples/sec', 
                                    i, loss, freq)
        except tf.errors.OutOfRangeError:
            break
    tf.logging.info('Epoch Training Time = %.3f', time() - epoch_start_time)

def tf_eval_loop(tf_sess, val_op):
    """ """
    # num_batches = 0
    # epoch_start_time = time()
    # val_acc = 0
    # while True:
    #     try:
    #         acc = tf_sess.run(val_op)
    #         val_acc += acc
    #         num_batches += 1
    #     except tf.errors.OutOfRangeError:
    #         break
    # if num_batches > 0:
    #     val_acc = (val_acc * 100.0) / num_batches

    # acc_summ = tf.summary.Summary()
    # summ_val = acc_summ.value.add()
    # summ_val.simple_value = val_acc
    # summ_val.tag = "Validation-Accuracy"
    # tb_writer.add_summary(acc_summ, self.epoch)
    # logger.info('Epoch Validation Time = %.3f', self.tick() - epoch_start_time)
    # self.logger.info('Epoch[%d] Validation-Accuracy = %.2f%%', self.epoch, val_acc)

def train(dataset="CIFAR-10"):
    """ """
    start_time = time()
    num_epoch  = 1
    with tf.Graph().as_default():
        batch_size = 200
        train_init_op, val_init_op, iter_op = tf_create_iterator(dataset, batch_size)
        images, labels = iter_op
        train_op, val_op, global_step, summ = tf_create_train_op(images, labels)

        # Create and initialize a TF Session
        tf_sess = tf.Session()
        tf_sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=tf_sess, coord=coord)

        # Create Tensorboard summary writer
        tb_writer = tf.summary.FileWriter("D:\\SNPX_ML\\python\\test_night", graph=tf_sess.graph)

        # Training Loop
        for epoch in range(num_epoch):
            # Training
            tf_sess.run(train_init_op)
            tf_train_loop(tf_sess, train_op, global_step, tb_writer, summ, batch_size)
            # Validation
            tf_sess.run(val_init_op)
            tf_eval_loop(tf_sess, val_op)
            # Flush Tensorboard Writer
            # tb_writer.flush()

        # tb_writer.close()
        coord.request_stop()
        coord.join(threads)
        tf_sess.close()

def trainyuyu():
    """ """
    tf.logging.set_verbosity(tf.logging.INFO)
    batch_size = 200
    dataset_prefix = os.path.join(SNPX_DATASET_ROOT, "CIFAR-10", "CIFAR-10")
    train_rec_file = dataset_prefix + "_train.tfrecords"

    with tf.Graph().as_default():
        # Create the training dataset object
        train_set = TFRecordDataset(train_rec_file)
        train_set = train_set.map(tf_parse_record, num_threads=4, output_buffer_size=1000)
        train_set = train_set.shuffle(buffer_size=10000)
        train_set = train_set.batch(batch_size)

        # Create a reinitializable iterator from both datasets
        iterator  = train_set.make_one_shot_iterator()
        images, labels = iterator.get_next()
        onehot_labels  = tf.one_hot(labels, 10)
        predictions    = snpx_net_create(10, images)

        # Get the optimizer
        opt = tf.train.AdamOptimizer()
        global_step = tf.train.get_or_create_global_step()
        
        # Compute the loss and the train_op
        loss = tf.losses.softmax_cross_entropy(onehot_labels, predictions)
        total_loss = tf.losses.get_total_loss()
        train_op   = opt.minimize(total_loss, global_step=global_step)
        op = [train_op, total_loss]
        tf_sess = tf.Session()
        tf_sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=tf_sess, coord=coord)

        count = 0
        last_log_tick = time()
        last_log_batch = 0
        while True:
            try:
                loss, step = tf_sess.run([op, global_step])
                count += 1
                print (count, step)
                if (count - last_log_batch) >= 10:
                    elapsed = time() - last_log_tick
                    freq = ((count - last_log_batch) * batch_size / elapsed)
                    last_log_batch = count
                    last_log_tick  = time()
                    print (count, loss, freq)
            except tf.errors.OutOfRangeError:
                break


