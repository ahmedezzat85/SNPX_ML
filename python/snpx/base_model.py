""" Synaplexus Base Model API 
"""
from __future__ import absolute_import

import os
import sys
import json
import logging
from time import time
from datetime import datetime
from importlib import import_module

from . util import snpx_create_dir, DictToAttrs

class SNPXModel(object):
    """ Base model
    The base class for all trainable neural network models.

    Parameters
    ----------
    model_name : str
        Name of the neural network architecture(e.g. VGG). It should match a file name containing the 
        model definition.
    dataset_name : str
        Name of the target dataset_name(e.g. `CIFAR-10`).
    backend_root : path
        Root path of the backend tool abstractions.
    num_class : int
        Number of classes of the dataset_name (For classification tasks).
    context : str
        A comma separeted list of the GPU IDs. For example, "0,1,2) means train on GPUS 0, 1, 2.
        If None, train on CPU.
    batch_size : int
        Training mini-batch size.
    fp16 : bool
        Whether to use FP16 for model representation.
    net_create : bool
        Whether to create the network architecture handler from an existing definition.
    """
    def __init__(self, model_name, dataset_name, backend, logs_root, model_bin_root):
        """
        """
        if not isinstance(model_name, str):
            raise ValueError("model_name is not a string")
        if not isinstance(dataset_name, str):
            raise ValueError("dataset_name is not a string")

        # Parameter initializations
        self.logger          = None
        self.model_fn        = None
        self.dataset_name    = dataset_name
        self.model_name      = model_name

        # Get the neural network model function
        net_module    = import_module('snpx.' + backend + '.arch.' + model_name)
        self.model_fn = net_module.snpx_net_create

        # Create the log directories
        date_time = datetime.now().strftime("%Y%m%d-%H.%M")
        self.log_dir = os.path.join(logs_root, backend, dataset_name, model_name, 'run_'+date_time)
        snpx_create_dir(self.log_dir)
        self.chkpt_prfx = os.path.join(self.log_dir, "chkpt") + self.model_name

        self.model_dir = os.path.join(model_bin_root, backend, dataset_name, model_name)
        snpx_create_dir(self.model_dir)
        self.model_prfx = os.path.join(self.model_dir, model_name)

        # Initialize the tick
        self.base_tick  = time()

    def tick(self):
        return time() - self.base_tick
    
    def _create_logger(self, log_file=None, mode='w'):
        """ Create a Logger Instance."""
        # Delete the old logger
        if self.logger is not None:
            for h in self.logger.handlers[:]:
                self.logger.removeHandler(h)

        # Create a new logger instance
        self.logger = logging.getLogger(self.model_name)
        self.logger.setLevel(logging.INFO)
        
        ## Add a console handler
        hdlr = logging.StreamHandler()
        hdlr.setFormatter(logging.Formatter(fmt="(%(name)s)[%(levelname)s]%(message)s"))
        self.logger.addHandler(hdlr)

        ## Add a file handler
        if log_file is not None:
            hdlr = logging.FileHandler(filename=log_file, mode=mode)
            hdlr.setFormatter(logging.Formatter(fmt="%(message)s"))
            self.logger.addHandler(hdlr)
        
        self.logger.propagate = False
            
    def train(self, 
              num_epoch, 
              batch_size, 
              optmz  = 'Adam', 
              lr     = 1e-3, 
              l2_reg = 0, 
              lr_decay = 0.9, 
              lr_decay_step = 3,
              log_every_n_batches = 10,
              log_every_n_sec=5):
        """ Train the network.
        """
        # Create a logger instance
        self._create_logger(log_file=os.path.join(self.log_dir, "Train.log"))

        # Pack the hyper-parameters
        hp_dict = {
            "batch_size" : batch_size,
            "optimizer"  : optmz,
            "lr"         : lr,
            "l2_reg"     : l2_reg,
            "lr_decay"   : lr_decay,
            "lr_step"    : lr_decay_step
        }
        self.hp = DictToAttrs(hp_dict)
        self.batch_size = self.hp.batch_size

        with open(os.path.join(self.log_dir, self.model_name + '-hp.json'), 'w') as fp:
            json.dump(hp_dict, fp, indent=0)

        self.log_freq   = log_every_n_sec
        self.batch_size = batch_size

        train_start_time  = datetime.now()
        self.logger.info("Training Started at  : " + train_start_time.strftime("%Y-%m-%d %H:%M:%S"))
        # Perform training in backend
        self.train_model(num_epoch)
        self.logger.info("Training Finished at : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.logger.info("Total Training Time  : " + str(datetime.now() - train_start_time))

    def evaluate(self, batch_size=256):
        """ """
        self._create_logger()        
        self.batch_size = batch_size
        start_time  = datetime.now()
        self.logger.info("Evaluating Model " + self.model_name + " on " + self.dataset_name)
        # Perform training in backend
        acc = self.evaluate_model()
        self.logger.info("Validation Accuracy = %.2F%%", acc)
        self.logger.info("Validation Time  : " + str(datetime.now() - start_time))
        
    def train_model(self, num_epoch):
        raise NotImplementedError()

    def evaluate_model(self):
        raise NotImplementedError()

    def resume_training(self, n_epoch):
        """
        """
        raise NotImplementedError()