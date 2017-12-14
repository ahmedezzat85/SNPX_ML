from __future__ import absolute_import

import os
import sys
import logging
from time import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

class SNPXTensorFlow(object):
    """ Class for training a deep learning model.
    """
    def __init__(self, 
                model_name, 
                dataset,
                data_format='NHWC',
                devices=['CPU'], 
                use_fp16=False,
                debug=False,
                data_aug=[], 
                extend_dataset=False,
                logs_root=None,
                model_bin_root=None):
        super(SNPXClassifier, self).__init__(model_name, dataset, "snpx_tf",
                                             logs_root, model_bin_root)
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

    def _forward_prop(self, batch, training=True):
        """ """
        return self.model_fn(self.num_classes, batch, self.data_format, is_training=training)

    def _create_train_op(self):
        """ """
        # Forward Propagation
        predictions = self._forward_prop(images)
        
        # Get the optimizer
        opt_str = self.hp['optimizer']
        if(opt_str.lower() == 'sgd'):
            opt = tf.train.GradientDescentOptimizer(learning_rate=self.hp['learning_rate'])
        else:
            opt = tf.train.AdamOptimizer(learning_rate=self.hp['learning_rate'])
        
        # Compute the loss and the train_op
        self.global_step_op = tf.train.get_or_create_global_step()
        update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            loss = tf.losses.softmax_cross_entropy(labels, predictions)
            self.total_loss = tf.losses.get_total_loss()
            self.train_op = opt.minimize(self.total_loss, self.global_step_op)
        self._create_eval_op(predictions, labels)

    def _create_eval_op(self, predictions, labels):
        """ """
        acc_tensor   = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))
        self.eval_op = tf.reduce_mean(tf.cast(acc_tensor, tf.float32))

    def _train_loop(self):
        """ """
        # Initialize the training Dataset Iterator
        self.tf_sess.run(self.train_set_init_op)

        epoch_start_time = self.tick()
        last_log_tick    = epoch_start_time
        last_step        = self.tf_sess.run(self.global_step_op)
        while True:
            try:
                _, loss, s, step = self.tf_sess.run([self.train_op, self.total_loss, 
                                                     self.summary_op, self.global_step_op])
                self.tb_writer.add_summary(s, step)
                self.tb_writer.flush()
                if (step - last_step) >= self.log_batch_freq:
                    elapsed = self.tick() - last_log_tick
                    freq = ((step - last_step)  * self.batch_size) / elapsed
                    last_step = step
                    last_log_tick  = self.tick()
                    self.logger.info('(%.3f)Epoch[%d] Batch[%d]\tloss: %.3f\tspeed: %.3f samples/sec', 
                                      self.tick(), self.epoch, step, loss, freq)
            except tf.errors.OutOfRangeError:
                break
        self.logger.info('Epoch Training Time = %.3f', self.tick() - epoch_start_time)
        self.saver.save(self.tf_sess, self.chkpt_prfx, self.epoch)

    def _eval_loop(self):
        """ """
        self.tf_sess.run(self.eval_set_init_op)
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
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.tf_sess, coord=self.coord)

    def train_model(self, num_epoch):
        """ """
        with tf.Graph().as_default():
            images, labels = self._load_dataset()
            self._create_train_op(images, labels)
            self.saver = tf.train.Saver()
            tf.add_to_collection('train_op', self.train_op)
            tf.add_to_collection('eval_op', self.eval_op)

            # Create a TF Session
            self.create_tf_session()

            # Create Tensorboard stuff
            self.summary_op = tf.summary.scalar("loss", self.total_loss)
            self.tb_writer  = tf.summary.FileWriter(self.log_dir, graph=self.tf_sess.graph)

            # tfdebug Hook
            if self.debug is True:
                self.tf_sess = tf_debug.LocalCLIDebugWrapperSession(self.tf_sess)
                self.tf_sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            # Training Loop
            for self.epoch in range(num_epoch):
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
            self.coord.request_stop()
            self.coord.join(self.threads)
            self.tf_sess.close()

    def evaluate_model(self):
        """ """
        with tf.Graph().as_default():
            # Load the Evaluation Dataset
            images, labels = self._load_dataset(training=False)

            # Forward Prop
            predictions = self._forward_prop(images, training=False)

            # Create a TF Session
            self.create_tf_session()
 
            # Load the saved model from a checkpoint
            chkpt_state = tf.train.get_checkpoint_state(self.model_dir)
            self.logger.info("Loading Checkpoint " + chkpt_state.model_checkpoint_path)
            tf_model = tf.train.Saver()
            tf_model.restore(self.tf_sess, chkpt_state.model_checkpoint_path)

            # Perform Model Evaluation
            self._create_eval_op(predictions, labels)
            acc = self._eval_loop()

            self.tf_sess.close()
        return acc

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
	# 	    graph_def.ParseFromString(f.read())
    #     tf.import_graph_def(graph_def, name="")

    def deploy_model(self, image_size=[32, 32]):
        """ """
        if not isinstance(image_size, list):
            raise ValueError('Invalid type <%s> for image_size it must be a list', type(image_size))

        with tf.Graph().as_default():
            # Create a placeholder
            shape = [1] + image_size + [3]
            image = tf.placeholder("float", shape, name='input')

            # Forward Prop
            out = self._forward_prop(image, training=False)
            prediction = tf.nn.softmax(out, name='output')

            # Create a TF Session
            self.create_tf_session()
 
            # Load the saved model from a checkpoint
            chkpt_state = tf.train.get_checkpoint_state(self.model_dir)
            self.logger.info("Loading Checkpoint " + chkpt_state.model_checkpoint_path)
            tf_model = tf.train.Saver()
            tf_model.restore(self.tf_sess, chkpt_state.model_checkpoint_path)

            tf_model_deploy = tf.train.Saver()
            tf_model_deploy.save(self.tf_sess, self.model_deploy_prfx)

            self.tf_sess.close()
        

