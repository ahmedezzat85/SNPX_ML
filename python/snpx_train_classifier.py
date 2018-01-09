""" Synaplexus Trainer Script 
"""
import os
import snpx
import numpy as np
from snpx_arg_parser import snpx_parse_cmd_line_options


def main():
    args = snpx_parse_cmd_line_options()
    classifier = snpx.get_classifier(args)
    classifier.train(num_epoch    = args.num_epoch, 
                    batch_size    = args.batch_size,
                    start_epoch   = args.begin_epoch,
                    optmz         = args.optimizer, 
                    lr            = args.lr, 
                    l2_reg        = args.l2_reg,
                    lr_decay      = args.lr_decay, 
                    lr_decay_step = args.lr_step)

def test():
    args = snpx_parse_cmd_line_options()
    # lr_list = [0.1, 0.09, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.005, 0.004, 0.001]
    
    for i in range(100):
        lr = 10**np.random.uniform(-4, -1)
        wd = 10**np.random.uniform(-5, -2)
        args.logs_subdir = 'mlp-' + str(i)
        print ('ITERATION = ', i, '   ===> ', lr, wd)
        classifier = snpx.get_classifier(args)
        classifier.train(num_epoch    = 10, 
                        batch_size    = 128,
                        start_epoch   = 0,
                        optmz         = 'adam', 
                        lr            = lr, 
                        l2_reg        = wd,
                        lr_decay      = args.lr_decay, 
                        lr_decay_step = args.lr_step)
        classifier.close()

if __name__ == '__main__':
    main()