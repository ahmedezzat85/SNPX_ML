""" mxnet classifier abstraction 
"""
from __future__ import absolute_import

import os
import sys
import json
import logging
import numpy as np
import mxnet as mx
import tensorboard
from time import time
from scipy import misc
from datetime import datetime
from importlib import import_module

from .. util import snpx_create_dir
from .. base_model import SNPXModel
from .mx_callback import EpochValCB, BatchEndCB
from .mx_dataset import mx_get_dataset_object

MXNET_ROOT = os.path.dirname(__file__)

def SNPX_CreateNetwork(net_name, num_class=10, use_fp16=False):
    """ Create the symbol from its name. The name must match the python
    file name containing the definition of this network symbol.
    """
    try:
        net_module  = import_module('snpx.snpx_mxnet.arch.' + net_name)
        return net_module.snpx_net_create(num_class, use_fp16)
    except ImportError:
        print ("ARCH Import Error", net_name)
        return None

class SNPXTrainer(SNPXModel):
    """ Class for training a deep learning model in mxnet.
    """
    def __init__(self, model_name, target_ds, batch_size=128, context=None, fp16=False, 
                    data_aug=False, net_create=True):
        if not isinstance(model_name, str):
            raise ValueError("model_name is not a string")
        if not isinstance(target_ds, str):
            raise ValueError("dataset is not a string")

        # Parameter Initializations
        self.hp             = {}
        self.logger         = None
        self.model_name     = model_name
        self.batch_size     = batch_size
        self.hp_initialized = False

        # Create the Model sub-dir (model/target_ds/model_name) which will
        # contain the trained model parameters
        self.model_dir   = os.path.join(MXNET_ROOT, 'model', target_ds, model_name)
        self.model_prfx  = os.path.join(self.model_dir, model_name)
        snpx_create_dir(self.model_dir)

        # Tensorboard stuff (For logging and visualization of training)
        ## Root Directory for Tensorboard logging for this model (logs/model_name)
        self.tb_root    = os.path.join(MXNET_ROOT, "logs", self.model_name)
        ## Directory holding the logs corresponding to the current iteration only(logs/model_name/%Date_Time%/)
        self.run_dir    = os.path.join(self.tb_root, datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        self.tb_writer  = tensorboard.SummaryWriter(self.run_dir)

        # Create Checkpoint directory
        chkpt_dir = os.path.join(self.run_dir, 'chkpt')
        snpx_create_dir(chkpt_dir)
        self.chkpt_prfx = os.path.join(chkpt_dir, 'CHKPT')

        # Load dataset
        self.dataset    = mx_get_dataset_object(target_ds, batch_size, data_aug)
        if net_create == True:
            self.Net = SNPX_CreateNetwork(net_name=model_name, num_class=self.dataset.n_class, use_fp16=fp16)

    def set_hyper_params(self, optmz='Adam', lr=1e-3, l2_reg=0, lr_decay=0.9, lr_decay_step = 3):
        """
        """
        # Initialize the Optimizer
        if(optmz.lower() == 'sgd'):
            self.optmz = mx.optimizer.SGD(learning_rate=lr, wd=l2_reg, rescale_grad=(1.0/self.batch_size), momentum=0.9)
        else:
            self.optmz = mx.optimizer.Adam(learning_rate=lr, wd=l2_reg, rescale_grad=(1.0/self.batch_size))
        
        # Create a logger instance
        self._create_logger(log_file=os.path.join(self.run_dir, "Train.log"))
        
        self.init           = init=mx.initializer.Xavier(magnitude=2.34, factor_type="in")
        self.val_acc        = []
        self.train_acc      = []
        self.batch_cb       = BatchEndCB(train_acc=self.train_acc, batch_size=self.batch_size, logger=self.logger)
        self.val_cb         = EpochValCB(self.optmz, self.val_acc, lr_decay, lr_decay_step, self.logger)
        self.hp_initialized = True

        # Log hyper parameters
        self.hp = {
            "Optimizer"       : optmz,
            "Batch Size"      : self.batch_size,
            "Learning rate"   : lr,
            "L2 Weight Decay" : l2_reg,
            "Lr Decay Factor" : lr_decay,
            "Lr Decay Step"   : lr_decay_step
        }
        with open(os.path.join(self.run_dir, 'train_param.json'), 'w') as fp:
            json.dump(self.hp, fp, indent=0)

        self.logger.info("[Training Setup]")
        self.logger.info("Batch Size      : %f", self.batch_size)
        self.logger.info("Optimizer       : %s", optmz)
        self.logger.info("Learning rate   : %f", lr)
        self.logger.info("L2 Weight Decay : %f", l2_reg)
        self.logger.info("Lr Decay Factor : %f", lr_decay)
        self.logger.info("Lr Decay Step   : %f", lr_decay_step)
        self.logger.info("-------------------------------------")

    def log_stats(self, name, value):
        """
        """
        sc_name = self.model_name+"/" + name
        n       = len(value)
        for i in range(n):
            self.tb_writer.add_scalar(sc_name, value[i], i)

    def viz_net_graph(self):
        """
        """
        shape = (1,) + self.dataset.data_shape
        g = mx.viz.plot_network(symbol=self.Net, title=self.model_name, shape={'data': shape}, save_format='png')
        g.render(filename=self.model_name, directory=self.run_dir)
        img = misc.imread(os.path.join(self.run_dir, self.model_name+".png"))
        self.tb_writer.add_image(self.model_name, img)

    def train(self, n_epoch):
        """ Train the network.
        """
        if self.hp_initialized is not True:
            raise ValueError("set_hyper_params must be called first")
        
        self.viz_net_graph()

        # Load training iterators
        train_iter, val_iter = self.dataset.load_train_data()
        start_time  = datetime.now()
        self.logger.info("Training Started at : " + start_time.strftime("%Y-%m-%d %H:%M:%S"))
        mx_module = mx.module.Module(symbol=self.Net, context=mx.gpu(0), logger=self.logger)
        chkpt_cb = mx.callback.module_checkpoint(mx_module, self.chkpt_prfx, save_optimizer_states=True)
        mx_module.fit(num_epoch=n_epoch, optimizer=self.optmz, initializer=self.init, 
                        train_data=train_iter, eval_data=val_iter, eval_end_callback=self.val_cb, 
                        batch_end_callback=self.batch_cb, epoch_end_callback=chkpt_cb)
        
        self.logger.info("Training Finished at : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.logger.info("Total Training Time  : " + str(datetime.now() - start_time))

        # Visualize learning
        self.log_stats("Train_Acc", self.train_acc)
        self.log_stats("Val_Acc", self.val_acc)
        self.tb_writer.close()

        # Save the model with the best validation accuracy
        best_epoch = self.val_acc.index(max(self.val_acc))
        _, args, auxs = mx.model.load_checkpoint(prefix=self.chkpt_prfx, epoch=(best_epoch + 1))
        mx_module.set_params(args, auxs)
        mx_module.save_checkpoint(self.model_prfx, 0)
        with open(self.model_prfx + '-hp.json', 'w') as fp:
            json.dump(self.hp, fp, indent=0)

    def resume_training(self, n_epoch):
        """
        """
        # load module
        # load the last saved hp for resuming 
        # 


    def score(self):
        """ Compute the test accuracy of a model
        """
        test_iter = self.dataset.load_test_data()
        
        # Create a logger instance
        self._create_logger(log_file="Evaluate.log")
        mx_module = mx.mod.Module.load(prefix=self.model_prfx, epoch=0, load_optimizer_states=False, 
                            context=mx.gpu(0), logger=self.logger)
        mx_module.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label, for_training=False)
        start_time  = datetime.now()
        self.logger.info("Testing Started at : " + start_time.strftime("%Y-%m-%d %H:%M:%S"))
        name_val = mx_module.score(eval_data=test_iter, eval_metric=mx.metric.Accuracy())
        elapsed     = datetime.now() - start_time
        score_time  = elapsed / test_iter.num_data
        self.logger.info("--------- Model \"%s\" Test Status ---------", self.model_name)
        for name, acc in name_val:
            self.logger.info("%s : %f", name, acc)
        self.logger.info("Total Score Time  : " + str(elapsed))
        self.logger.info("Score Time = " + str(score_time))

        return acc

class SNPXFineTuner(SNPXTrainer):
    """ mxnet model abstraction
    """
    def __init__(self, modelname, base_dataset, target_ds, batch_size=128, context='GPU', data_aug=False):
        if not isinstance(base_dataset, str):
            raise ValueError("base_dataset is not a string")
        
        super(SNPXFineTuner, self).__init__(modelname, target_ds, batch_size, context, net_create=False)
        self.mx_symbol      = None
        self.model_arg      = None
        self.model_aux      = None
        self.fixed_args     = None
        self.dataset        = mx_get_dataset_object(target_ds, self.batch_size, data_aug)
        self.num_class      = self.dataset.n_class

        # Load the pretrained model
        pretr_prfx = os.path.join(BACKEND_DIR, 'model', base_dataset, self.model_name, self.model_name)
        self.mx_symbol , self.model_arg, self.model_aux = mx.model.load_checkpoint(pretr_prfx, 0)
        self.modify_net('flatten0_out')

    def modify_net(self, last_layer):
        """
        """
        if not isinstance(last_layer, str):
            raise ValueError("last_layer is not a string")
        if not last_layer.endswith('_output'):
            last_layer += "_output"
        
        # Search for the last_layer to cut out the network at it
        sym_internals   = self.mx_symbol.get_internals()
        layer_names     = sym_internals.list_outputs()[::-1]
        layer_outputs   = [out for out in layer_names if out.endswith('output')]
        if last_layer not in layer_outputs:
            raise ValueError("%s not found", last_layer)
        
        # Append a new classifier output layer
        fc_in           = sym_internals[last_layer]
        fc              = mx.sym.FullyConnected(data=fc_in, num_hidden=self.dataset.n_class)
        self.mx_symbol  = mx.sym.SoftmaxOutput(data=fc, name='softmax')

    def set_frozen_layers(self, last_frozen=None):
        """ Freeze the first few layers in a pre-trained model
        """
        if last_frozen is None:
            return

        arg_names   = self.mx_symbol.list_arguments()
        if last_frozen not in arg_names:
            print arg_names
            raise ValueError("Incorrect value for last_frozen_layer. Not found.")
        idx = arg_names.index(last_frozen) + 1
        self.fixed_args = arg_names[0:idx]
        for name, val in aux_param:
            self.fixed_args.append(name)

    def train(self, n_epoch):
        """ Train the network.
        """
        if self.hp_initialized is not True:
            raise ValueError("set_hyper_params must be called first")
        train_iter, val_iter = self.dataset.load_train_data()
        start_time  = datetime.now()
        self.logger.info("Training Started at : " + start_time.strftime("%Y-%m-%d %H:%M:%S"))
        mx_module = mx.module.Module(symbol=self.mx_symbol, context=mx.gpu(0), logger=self.logger, 
                                        fixed_param_names=self.fixed_args)
        chkpt_cb = mx.callback.module_checkpoint(mx_module, self.chkpt_prfx, save_optimizer_states=True)
        mx_module.fit(num_epoch=n_epoch, optimizer=self.optmz, initializer=self.init, 
                        train_data=train_iter, eval_data=val_iter, arg_params=self.model_arg,
                        aux_params=self.model_aux, allow_missing=True, eval_end_callback=self.val_cb, 
                        batch_end_callback=self.batch_cb, epoch_end_callback=chkpt_cb)
        
        self.logger.info("Training Finished at : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.logger.info("Total Training Time  : " + str(datetime.now() - start_time))

        # Save the model with the best validation accuracy
        best_epoch = self.val_acc.index(max(self.val_acc))
        _, args, auxs = mx.model.load_checkpoint(prefix=self.chkpt_prfx, epoch=(best_epoch + 1))
        mx_module.set_params(args, auxs)
        mx_module.save_checkpoint(self.model_prfx, 0)
        np.array(self.val_acc).tofile(os.path.join(self.model_dir, 'val_acc.list'), sep=',')
        np.array(self.train_acc).tofile(os.path.join(self.model_dir, 'train_acc.list'), sep=',')