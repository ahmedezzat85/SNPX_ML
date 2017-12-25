from __future__ import absolute_import

import os
import sys
import logging
from time import time
from datetime import datetime

import numpy as np
import mxnet as mx
from scipy import misc

from .. base_model import SNPXModel
from . mx_callback import EpochValCB, BatchEndCB, TensorboardWriter
from . mx_dataset import MxDataset

class SNPXMxnetClassifier(SNPXModel):
    """ Class for training a deep learning model.
    """
    def __init__(self, 
                model_name, 
                dataset_name, 
                devices=['CPU'], 
                use_fp16=False, 
                data_aug=False, 
                extend_dataset=False,
                logs_root=None,
                logs_subdir=None,
                model_bin_root=None):
        super(SNPXMxnetClassifier, self).__init__(model_name, dataset_name, "snpx_mxnet", 
                                                    logs_root, model_bin_root, logs_subdir)
        self.symbol = None
        self.data_aug = data_aug

    def viz_net_graph(self):
        """
        """
        shape = (1,) + self.dataset.data_shape
        g = mx.viz.plot_network(symbol=self.symbol, title=self.model_name, shape={'data': shape}, 
                                    save_format='png')
        g.render(filename=self.model_name, directory=self.log_dir)
        img = misc.imread(os.path.join(self.log_dir, self.model_name+".png"))

    def train_model(self, num_epoch, begin_epoch=0):
        """ """
        if self.data_aug is True: self.logger.info('Using Data Augmentation')

        # Initialize the Optimizer
        opt = self.hp.optimizer.lower()
        # print (self.hp.l2_reg)
        opt_param = (('learning_rate', self.hp.lr), ('wd', self.hp.l2_reg),)
        if  opt == 'sgd': opt_param += (('momentum', 0.9),)

        # Load dataset
        self.dataset = MxDataset(self.dataset_name, self.batch_size, data_aug=self.data_aug)
        if begin_epoch == 0:
            self.symbol = self.model_fn(self.dataset.num_classes)
            mx_module = mx.module.Module(symbol=self.symbol, context=mx.gpu(0), logger=self.logger)
            resume = False
        else:
            resume = True
            mx_module = mx.module.Module.load(self.chkpt_prfx, begin_epoch, context=mx.gpu(0), 
                                                logger=self.logger)

        self.viz_net_graph()

        # Load training iterators
        tb_writer     = TensorboardWriter(self.log_dir, reuse=resume)
        self.init     = init=mx.initializer.Xavier(magnitude=2.34, factor_type="in")
        self.batch_cb = BatchEndCB(tb_writer, self.batch_size, logger=self.logger)
        self.val_cb   = EpochValCB(tb_writer, self.logger)
        chkpt_cb = mx.callback.module_checkpoint(mx_module, self.chkpt_prfx, save_optimizer_states=False)
        mx_module.fit(train_data=self.dataset.mx_train_iter, eval_data=self.dataset.mx_eval_iter, 
                      epoch_end_callback=chkpt_cb, batch_end_callback=self.batch_cb,
                      optimizer=opt, optimizer_params=opt_param, eval_end_callback=self.val_cb,
                      initializer=self.init, num_epoch=num_epoch)
        
        self.val_cb(None)