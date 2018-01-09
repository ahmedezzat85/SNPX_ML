from __future__ import absolute_import

import os
import sys
import logging
from time import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from .. base_model import SNPXModel
from . tf_dataset import TFDataset

class SNPXTensorflowClassifier(SNPXModel):
    """ Class for training a deep learning model.
    """
    def __init__(self, 
                 model_name, 
                 dataset_name,
                 data_format='NHWC',
                 devices=['CPU'], 
                 use_fp16=False,
                 debug=False,
                 data_aug=False, 
                 logs_root=None,
                 logs_subdir=None,
                 model_bin_root=None):
        super().__init__(model_name, dataset_name, "snpx_tf", logs_root, model_bin_root, logs_subdir)
        # Disable Tensorflow logs except for errors
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Initialization
        self.dtype       = tf.float32 if use_fp16 == False else tf.float16
        self.debug       = debug
        self.tf_sess     = None
        self.eval_op     = None
        self.loss        = None
        
        self.train_op    = None
        self.summary_op  = None
        self.data_format = data_format
        self.global_step = None
        self.data_aug    = data_aug

    def _load_dataset(self, training=True):
        """ """
        with tf.device('/cpu:0'):
            self.dataset = TFDataset(self.dataset_name, self.batch_size, training, self.dtype,
                                        self.data_format, self.data_aug)

    def _forward_prop(self, batch, num_classes, training=True):
        """ """
        logits, predictions = self.model_fn(num_classes, batch, self.data_format, is_training=training)
        return logits, predictions

    def _create_train_op(self, logits):
        """ """
        self.global_step = tf.train.get_or_create_global_step()

        # Get the optimizer
        if self.hp.lr_decay:
            lr = tf.train.exponential_decay(self.hp.lr, self.global_step, 1000, 0.94, True)
        else:
            lr = self.hp.lr
        tf.summary.scalar("Learning Rate", lr)

        optmz = self.hp.optimizer.lower()
        if optmz == 'sgd':
            opt = tf.train.MomentumOptimizer(lr, momentum=0.9)
        elif optmz == 'adam':
            eps = 1e-8 if self.dtype is tf.float32 else 1e-4
            opt = tf.train.AdamOptimizer(lr, epsilon=eps)
        elif optmz == 'rmsprop':
            eps = 1e-8
            opt = tf.train.RMSPropOptimizer(lr, epsilon=eps)
 
        # Compute the loss and the train_op
        cross_entropy = tf.losses.softmax_cross_entropy(self.dataset.labels, logits) # needs wrapping
        self.loss = cross_entropy
        if self.hp.l2_reg > 0:
            l2_loss = self.hp.l2_reg * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = self.loss + l2_loss
        tf.summary.scalar("Cross Entropy", cross_entropy)

        update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = opt.minimize(self.loss, self.global_step)

        self._create_eval_op(logits, self.dataset.labels)

    def _create_eval_op(self, predictions, labels):
        """ """
        # self.eval_op = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions)
        acc_tensor   = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))
        self.eval_op = tf.reduce_mean(tf.cast(acc_tensor, tf.float32))

    def _train_loop(self):
        """ """
        # Initialize the training Dataset Iterator
        self.tf_sess.run(self.dataset.train_set_init_op)

        epoch_start_time = self.tick()
        last_log_tick    = epoch_start_time
        last_step        = self.tf_sess.run(self.global_step)
        while True:
            try:
                feed_dict = {self.training: True}
                fetches   = [self.loss, self.train_op, self.summary_op, self.global_step]
                loss, _, s, step = self.tf_sess.run(fetches, feed_dict)
                self.tb_writer.add_summary(s, step)
                self.tb_writer.flush()
                elapsed = self.tick() - last_log_tick
                if elapsed >= self.log_freq:
                    speed = ((step - last_step)  * self.batch_size) / elapsed
                    last_step = step
                    last_log_tick  = self.tick()
                    self.logger.info('(%.3f)Epoch[%d] Batch[%d]\tloss: %.3f\tspeed: %.3f samples/sec', 
                                      self.tick(), self.epoch, step, loss, speed)
            except tf.errors.OutOfRangeError:
                break
        self.logger.info('Epoch Training Time = %.3f', self.tick() - epoch_start_time)
        self.saver.save(self.tf_sess, self.chkpt_prfx, self.epoch)

    def _eval_loop(self):
        """ """
        self.tf_sess.run(self.dataset.eval_set_init_op)
        epoch_start_time = self.tick()
        val_acc = 0
        n = 0
        while True:
            try:
                feed_dict = {self.training: False}
                batch_acc = self.tf_sess.run(self.eval_op, feed_dict)
                val_acc += batch_acc
                n += 1
            except tf.errors.OutOfRangeError:
                break
        eval_acc = (val_acc * 100.0) / n
        self.logger.info('Validation Time = %.3f', self.tick() - epoch_start_time)
        return eval_acc

    def create_tf_session(self):
        """ """
        # Session Configurations 
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 4
        config.gpu_options.allow_growth = True # Very important to avoid OOM errors
        config.gpu_options.per_process_gpu_memory_fraction = 1.0 #0.4

        # Create and initialize a TF Session
        self.tf_sess = tf.Session(config=config)
        self.tf_sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    def train_model(self, num_epoch, begin_epoch=0):
        """ """
        with tf.Graph().as_default():
            self._load_dataset()

            # Forward Propagation
            self.training = tf.placeholder(tf.bool, name='Train_Flag')
            logits, _ = self._forward_prop(self.dataset.images, self.dataset.num_classes, self.training)
        
            self._create_train_op(logits)

            # Create a TF Session
            self.create_tf_session()

            # Create Tensorboard stuff
            self.summary_op = tf.summary.merge_all()
            self.tb_writer  = tf.summary.FileWriter(self.log_dir, graph=self.tf_sess.graph)

            if begin_epoch > 0:
                # Load the saved model from a checkpoint
                chkpt = self.chkpt_prfx + '-' + str(begin_epoch)
                self.logger.info("Loading Checkpoint " + chkpt)
                begin_epoch += 1
                num_epoch   += begin_epoch
                self.saver = tf.train.Saver(max_to_keep=200)
                self.saver.restore(self.tf_sess, chkpt)
                self.tb_writer.reopen()
            else:
                self.saver = tf.train.Saver(max_to_keep=200)

            # tfdebug Hook
            if self.debug is True:
                self.tf_sess = tf_debug.LocalCLIDebugWrapperSession(self.tf_sess)
                self.tf_sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            # Training Loop
            for self.epoch in range(begin_epoch, num_epoch):
                # Training
                self._train_loop()
                # Validation
                val_acc = self._eval_loop()
                # Visualize Training
                acc_summ = tf.summary.Summary()
                summ_val = acc_summ.value.add(simple_value=val_acc, tag="Validation-Accuracy")
                self.tb_writer.add_summary(acc_summ, self.epoch)
                self.logger.info('Epoch[%d] Validation-Accuracy = %.2f%%', self.epoch, val_acc)
                # Flush Tensorboard Writer
                self.tb_writer.flush()

            # Save the last checkpoint
            self.saver.save(self.tf_sess, self.model_prfx)
            
            # Close and terminate
            self.tb_writer.close()
            self.tf_sess.close()

    def evaluate_model(self):
        """ """
        with tf.Graph().as_default():
            # Load the Evaluation Dataset
            self._load_dataset(training=False)

            # Forward Prop
            self.training = tf.placeholder(tf.bool, name='Train_Flag')
            predictions, _ = self._forward_prop(self.dataset.images, self.dataset.num_classes, False)

            # Create a TF Session
            self.create_tf_session()
 
            # Load the saved model from a checkpoint
            chkpt_state = tf.train.get_checkpoint_state(self.model_dir)
            self.logger.info("Loading Checkpoint " + chkpt_state.model_checkpoint_path)
            tf_model = tf.train.Saver()
            tf_model.restore(self.tf_sess, chkpt_state.model_checkpoint_path)

            # Perform Model Evaluation
            self._create_eval_op(predictions, self.dataset.labels)
            acc = self._eval_loop()

            self.tf_sess.close()
        return acc

    def deploy(self, img_size, num_classes):
        """ """
        self._create_logger()        
        with tf.Graph().as_default():
            if self.data_format.startswith('NC'):
                in_shape = [1, 3, img_size, img_size]
            else:
                in_shape = [1, img_size, img_size, 3]
            input_image = tf.placeholder(self.dtype, in_shape, name='input_image')

            # Forward Prop
            logits, predictions = self._forward_prop(input_image, num_classes, 
                                                     predict=True, training=False)

            # Create a TF Session
            self.create_tf_session()
 
            # Load the saved model from a checkpoint
            chkpt_state = tf.train.get_checkpoint_state(self.model_dir)
            self.logger.info("Loading Checkpoint " + chkpt_state.model_checkpoint_path)
            tf_model = tf.train.Saver()
            tf_model.restore(self.tf_sess, chkpt_state.model_checkpoint_path)

            saver = tf.train.Saver()
            saver.save(self.tf_sess, self.deploy_prfx)
            self.tf_sess.close()

    # def evaluate_model__(self):
    #     """ """
    #     # Load the Evaluation Dataset
    #     self._load_dataset(training=False)

    #     # Forward Prop
    #     predictions = self._forward_prop(self.dataset.images, training=False)

    #     # Create a TF Session
    #     self.create_tf_session()

    #     # Load the saved model from a checkpoint
    #     chkpt_state = tf.train.get_checkpoint_state(self.model_dir)
    #     self.logger.info("Loading Checkpoint " + chkpt_state.model_checkpoint_path)
    #     tf_model = tf.train.Saver()
    #     tf_model.restore(self.tf_sess, chkpt_state.model_checkpoint_path)

    #     # Perform Model Evaluation
    #     self._create_eval_op(predictions, self.dataset.labels)
    #     acc = self._eval_loop()

    #     self.tf_sess.close()
    #     return acc

    def save_tf_graph(self):
        """ """
        if self.tf_sess is not None:
            tf.train.write_graph(self.tf_sess.graph_def, 
                                logdir=self.model_dir, 
                                name=self.model_name+'.pb', 
                                as_text=False)

    # def load_tf_graph(self):
    #     """ """
    #     graph_file = self.model_prfx + '.pb'
    #     graph_def = tf.GraphDef()
    #     with tf.gfile.FastGFile(graph_file, "rb") as f:
	# 		graph_def.ParseFromString(f.read())
		
	# 	tf.import_graph_def(
	# 		graph_def,
	# 		name=""
	# 	)

