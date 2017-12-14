from __future__ import absolute_import

import mxnet as mx
from time import time
import logging

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
    def __init__(self, batch_size, train_acc, frequent=10, logger=logging):
        self.batch_sz   = batch_size
        self.frequent   = frequent
        self.init       = False
        self.tic        = 0
        self.last_count = 0
        self.logger     = logger
        self.train_acc  = train_acc

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_sz / (time() - self.tic)
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    param.eval_metric.reset()
                    for name, value in name_value:
                        self.logger.info('Epoch[%d] Batch [%03d]\tSpeed: %.2f samples/sec\tTrain-%s=%f',
                                     param.epoch, count, speed, name, value)
                        if len(self.train_acc) == param.epoch:
                            self.train_acc.append(value)
                        else:
                            self.train_acc[param.epoch] = value
                else:
                    self.logger.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, speed)
                self.tic = time()
        else:
            self.init = True
            self.tic = time()

class EpochValCB(object):
    """ Epoch Validation End callback

    Called at the end of each epoch to 
    communicate the Validation results
    """
    def __init__(self, optmz, val_acc, lr_decay=1, lr_step=5, logger=logging):
        self.val_acc    = val_acc
        self.optmz      = optmz
        self.lr_decay   = lr_decay
        self.lr_step    = lr_step
        self.logger     = logger
        # Sanity Checks
        if self.lr_step == 0:
            self.lr_decay = 1
        if self.lr_decay == 0:
            self.lr_decay = 1
            
    def __call__(self, param):
        if not param.eval_metric:
            return
        name_value = param.eval_metric.get_name_value()
        name, value  = name_value[0]
        if(((param.epoch + 1) >= self.lr_step) and (((param.epoch + 1) % self.lr_step) == 0)):
            self.optmz.lr *= self.lr_decay
            self.logger.info('Switch LR to %f', self.optmz.lr)
        self.val_acc.append(value)


