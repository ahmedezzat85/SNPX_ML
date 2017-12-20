from __future__ import absolute_import

import os
import sys
import logging
from time import time
from datetime import datetime

import numpy as np
import mxnet as mx
import tensorboard
from scipy import misc

from .. base_model import SNPXModel
from . mx_callback import EpochValCB, BatchEndCB
from . mx_dataset import MxDataset
from .. util import snpx_create_dir

class SNPXMxnetClassifier(SNPXModel):
    """ Class for training a deep learning model.
    """
    def __init__(self, 
                model_name, 
                dataset_name, 
                devices=['CPU'], 
                use_fp16=False, 
                data_aug=[], 
                extend_dataset=False,
                logs_root=None,
                model_bin_root=None):
        super().__init__(model_name, dataset_name, "snpx_mxnet", logs_root, model_bin_root)
        self.symbol = None

    def log_stats(self, name, value):
        """
        """
        sc_name = self.model_name+"/" + name
        n       = len(value)
        # for i in range(n):
        #     self.tb_writer.add_scalar(sc_name, value[i], i)

    def viz_net_graph(self):
        """
        """
        shape = (1,) + self.dataset.data_shape
        g = mx.viz.plot_network(symbol=self.symbol, title=self.model_name, shape={'data': shape}, save_format='png')
        g.render(filename=self.model_name, directory=self.log_dir)
        img = misc.imread(os.path.join(self.log_dir, self.model_name+".png"))
        # self.tb_writer.add_image(self.model_name, img)

    def train_model(self, num_epoch):
        """ """
        # Initialize the Optimizer
        if(self.hp.optimizer.lower() == 'sgd'):
            self.optmz = mx.optimizer.SGD(learning_rate=self.hp.lr, 
                                          rescale_grad=(1.0/self.batch_size), momentum=0.9)
        else:
            self.optmz = mx.optimizer.Adam(learning_rate=self.hp.lr, 
                                           rescale_grad=(1.0/self.batch_size))

        self.init           = init=mx.initializer.Xavier(magnitude=2.34, factor_type="in")
        self.val_acc        = []
        self.train_acc      = []
        self.batch_cb       = BatchEndCB(train_acc=self.train_acc, batch_size=self.batch_size, 
                                         logger=self.logger)
        self.val_cb         = EpochValCB(self.optmz, self.val_acc, self.log_dir, self.logger)

        # Load dataset
        self.dataset = MxDataset(self.dataset_name, self.batch_size)
        self.symbol  = self.model_fn(self.dataset.num_classes)

        # self.tb_writer  = tensorboard.SummaryWriter(self.log_dir)
        self.viz_net_graph()


        # Create Checkpoint directory
        chkpt_dir = os.path.join(self.log_dir, 'chkpt')
        snpx_create_dir(chkpt_dir)
        self.chkpt_prfx = os.path.join(chkpt_dir, 'CHKPT')

        # Load training iterators
        start_time  = datetime.now()
        mx_module = mx.module.Module(symbol=self.symbol, context=mx.gpu(0), logger=self.logger)
        chkpt_cb = mx.callback.module_checkpoint(mx_module, self.chkpt_prfx, save_optimizer_states=True)
        mx_module.fit(num_epoch=num_epoch, 
                      optimizer=self.optmz, 
                      initializer=self.init, 
                      train_data=self.dataset.mx_train_iter, 
                      eval_data=self.dataset.mx_eval_iter, 
                      eval_end_callback=self.val_cb, 
                      batch_end_callback=self.batch_cb, 
                      epoch_end_callback=chkpt_cb)
        
        # Visualize learning
        self.log_stats("Training-Accuracy", self.train_acc)
        self.log_stats("Validation-Accuracy", self.val_acc)
        self.val_cb(None)

        # Save the model with the best validation accuracy
        best_epoch = self.val_acc.index(max(self.val_acc))
        _, args, auxs = mx.model.load_checkpoint(prefix=self.chkpt_prfx, epoch=(best_epoch + 1))
        mx_module.set_params(args, auxs)
        mx_module.save_checkpoint(self.model_bin_dir+self.model_name, 0)