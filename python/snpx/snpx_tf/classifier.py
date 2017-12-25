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
import tensorflow.contrib.slim as slim

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
                data_aug=[], 
                extend_dataset=False,
                logs_root=None,
                logs_subdir=None,
                model_bin_root=None):
        super().__init__(model_name, dataset_name, "snpx_tf",
                        logs_root, model_bin_root, logs_subdir)
        # Disable Tensorflow logs except for errors
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Initialization
        self.dtype          = tf.float32 if use_fp16 == False else tf.float16
        self.debug          = debug
        self.tf_sess        = None
        self.eval_op        = None
        self.train_op       = None
        self.total_loss     = None
        self.summary_op     = None
        self.data_format    = data_format
        self.global_step_op = None

    def _load_dataset(self, training=True):
        """ """
        self.dataset = TFDataset(self.dataset_name, self.batch_size, training, 
                                    self.dtype, self.data_format)

    def _forward_prop(self, batch, num_classes, predict=False, training=True):
        """ """
        predictions = None
        logits = self.model_fn(num_classes, batch, self.data_format, is_training=training)
        if predict is True:
            predictions = tf.nn.softmax(logits, name='Predictions')
        return logits, predictions

    def _create_train_op(self):
        """ """
        # Forward Propagation
        logits, predictions = self._forward_prop(batch=self.dataset.images, 
                                                 num_classes=self.dataset.num_classes,
                                                 predict=True,
                                                 training=True)
        
        # Get the optimizer
        if(self.hp.optimizer.lower() == 'sgd'):
            opt = tf.train.MomentumOptimizer(learning_rate=self.hp.lr, momentum=0.9)
        else:
            opt = tf.train.AdamOptimizer(learning_rate=self.hp.lr)
        
        # Compute the loss and the train_op
        self.global_step_op = tf.train.get_or_create_global_step()
        update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            loss = tf.losses.softmax_cross_entropy(self.dataset.labels, logits) # needs wrapping
            self.total_loss = tf.losses.get_total_loss()
            self.train_op = opt.minimize(self.total_loss, self.global_step_op)
        self._create_eval_op(predictions, self.dataset.labels)

    def _create_eval_op(self, predictions, labels):
        """ """
        acc_tensor   = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))
        self.eval_op = tf.reduce_mean(tf.cast(acc_tensor, tf.float32))

    def _train_loop(self):
        """ """
        # Initialize the training Dataset Iterator
        self.tf_sess.run(self.dataset.train_set_init_op)

        epoch_start_time = self.tick()
        last_log_tick    = epoch_start_time
        last_step        = self.tf_sess.run(self.global_step_op)
        while True:
            try:
                _, loss, s, step = self.tf_sess.run([self.train_op, self.total_loss, 
                                                     self.summary_op, self.global_step_op])
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
                batch_acc = self.tf_sess.run(self.eval_op)
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
            self._create_train_op()
            self.saver = tf.train.Saver()
            tf.add_to_collection('train_op', self.train_op)
            tf.add_to_collection('eval_op', self.eval_op)

            # Create a TF Session
            self.create_tf_session()
            # Create Tensorboard stuff
            self.summary_op = tf.summary.scalar("loss", self.total_loss)
            self.tb_writer  = tf.summary.FileWriter(self.log_dir, graph=self.tf_sess.graph)

            if begin_epoch > 0:
                # Load the saved model from a checkpoint
                chkpt_state = tf.train.get_checkpoint_state(self.chkpt_dir)
                self.logger.info("Loading Checkpoint " + chkpt_state.model_checkpoint_path)
                out = chkpt_state.model_checkpoint_path.strip().split('CHKPT-')
                begin_epoch = int(out[-1]) + 1
                tf_model = tf.train.Saver()
                tf_model.restore(self.tf_sess, chkpt_state.model_checkpoint_path)
                self.tb_writer.reopen()

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
            predictions = self._forward_prop(self.dataset.images, training=False)

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

