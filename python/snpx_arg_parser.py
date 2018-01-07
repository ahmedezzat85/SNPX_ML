""" Synaplexus Trainer Script 
"""
import os
import argparse

def snpx_parse_cmd_line_options():
    """
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--backend'         , type=str      , help='Backend DeepLearning Framework. Possible Values are: TensorFlow and mxnet (case insensitive')
    arg_parser.add_argument('--model-name'      , type=str      , help='Neural Network Model name. e.g. VGG.')
    arg_parser.add_argument('--gpu-list'        , type=str      , help='Comma separated list of gpu numbers. If this not given, default to CPU.')
    arg_parser.add_argument('--lr'              , type=float    , help='Learning Rate.')
    arg_parser.add_argument('--l2-reg'          , type=float    , help='L2 Regularization Parameter or Weight Decay.')
    arg_parser.add_argument('--optimizer'       , type=str      , help='Optimization technique for parameter update.')
    arg_parser.add_argument('--target-dataset'  , type=str      , help='Target training dataset.')
    arg_parser.add_argument('--base-dataset'    , type=str      , help='Base dataset this model was trained on (in case ofg fine-tuning a model).')
    arg_parser.add_argument('--use-fp16'        , type=int      , help='Whether to use fp16 for the entire model parameters. Default is fp32.')
    arg_parser.add_argument('--data-aug'        , type=int      , help='Use data-augmentation to extend the dataset.')
    arg_parser.add_argument('--batch-size'      , type=int      , help='Training mini-batch size.')
    arg_parser.add_argument('--lr-step'         , type=int      , help='Learning rate decay step in epochs.')
    arg_parser.add_argument('--lr-decay'        , type=int      , help='Learning rate decay rate.')
    arg_parser.add_argument('--data-format'     , type=str      , help='Data Format')
    arg_parser.add_argument('--num-epoch'       , type=int      , help='Number of epochs for the training process.')
    arg_parser.add_argument('--begin-epoch'     , type=int      , help='Epoch ID of from which the training process will start. Useful for training resume.')
    arg_parser.add_argument('--debug'           , type=int      , help='Use TF CLI debug')
    arg_parser.add_argument('--logs-dir'        , type=str      , help='Logs Directory')
    arg_parser.add_argument('--logs-subdir'     , type=str      , help='Logs Directory')
    arg_parser.add_argument('--bin-dir'         , type=str      , help='Logs Directory')

    arg_parser.set_defaults(
        backend         = 'tensorflow',
        target_dataset  = "CIFAR-10",
        model_name      = 'mlp',
        data_format     = 'NCHW',
        gpu_list        = '0',
        lr              = 1e-3,
        l2_reg          = 0,
        optimizer       = 'Adam',
        use_fp16        = 0,
        data_aug        = 0,
        batch_size      = 128,
        lr_step         = 10000,
        lr_decay        = 0,
        begin_epoch     = 0,
        num_epoch       = 1,
        debug           = 0,
        logs_dir        = os.path.join(os.path.dirname(__file__), "log"),
        logs_subdir     = '',
        bin_dir         = os.path.join(os.path.dirname(__file__), "model")
    )

    args = arg_parser.parse_args()
    return args