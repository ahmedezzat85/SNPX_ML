from __future__ import absolute_import

import mxnet as mx
from time import time
import logging

try:
    import tensorflow as tf
    class TensorboardWriter(object):
        def __init__(self, log_dir, reuse=False):
            self.tb_writer = tf.summary.FileWriter(log_dir)
            if reuse is True: self.tb_writer.reopen()

        def write_scalar(self, name, val, idx):
            # Write to Tensorboard
            acc_summ = tf.summary.Summary()
            summ_val = acc_summ.value.add(simple_value=val, tag=name)
            self.tb_writer.add_summary(acc_summ, idx)
            # Flush Tensorboard Writer
            self.tb_writer.flush()

        def close(self):
            self.tb_writer.close()

except ImportError:
    try:
        import tensorboard
        class TensorboardWriter(object):
            def __init__(self, log_dir):
                self.tb_writer  = tensorboard.SummaryWriter(log_dir)

            def write_scalar(self, name, val, idx):
                self.tb_writer.add_scalar(name, val, idx)
                self.tb_writer.flush()

            def close(self):
                self.tb_writer.close()

    except ImportError:
        class TensorboardWriter(object):
            def __init__(self, log_dir):
                pass

            def write_scalar(self, name, val, idx):
                pass

            def close(self):
                pass


class BatchEndCB(object):
    """Calculate and log training speed periodically.

    Parameters
    ----------
    batch_size: int
        batch_size of data
    frequent: int
        How many batches between calculations.
        Defaults to calculating & logging every 50 batches.
    """
    def __init__(self, tensorboard_hdl, batch_size, log_freq=1, logger=logging):
        self.batch_size = batch_size
        self.log_freq   = log_freq
        self.init       = False
        self.tic        = time()
        self.last_batch = 0
        self.logger     = logger
        self.tb_writer  = tensorboard_hdl

    def __call__(self, param):
        """Callback to Show speed."""
        batch = param.nbatch
        elapsed = time() - self.tic
        if elapsed >= self.log_freq:
            speed = ((batch - self.last_batch)  * self.batch_size) / elapsed
            if self.last_batch <= batch:
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    param.eval_metric.reset()
                    for name, value in name_value:
                        self.logger.info('Epoch[%d] Batch [%03d]\tSpeed: %.2f samples/sec\tTrain-%s=%f',
                                            param.epoch, batch, speed, name, value * 100)
                else:
                    self.logger.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                        param.epoch, batch, speed)
            self.tb_writer.write_scalar("Training-Accuracy", value, batch)
            self.tic = time()
            self.last_batch = batch

class EpochValCB(object):
    """ Epoch Validation End callback

    Called at the end of each epoch to 
    communicate the Validation results
    """
    def __init__(self, tensorboard_hdl, logger=logging):
        self.logger     = logger
        self.tb_writer  = tensorboard_hdl
            
    def __call__(self, param):
        if param is None:
            self.tb_writer.close()
            return

        if not param.eval_metric:
            return
        name_value = param.eval_metric.get_name_value()
        name, value  = name_value[0]
        value = value * 100
        self.tb_writer.write_scalar("Validation-Accuracy", value, param.epoch)


