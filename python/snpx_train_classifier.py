""" Synaplexus Trainer Script 
"""
import os
import snpx
from snpx_arg_parser import snpx_parse_cmd_line_options


def main():
    args = snpx_parse_cmd_line_options()
    classifier = snpx.get_classifier(args)
    classifier.train(num_epoch  = args.num_epoch, 
                    batch_size = args.batch_size,
                    optmz      = args.optimizer, 
                    lr         = args.lr, 
                    l2_reg     = args.l2_reg,
                    lr_decay   = args.lr_decay, 
                    lr_decay_step = args.lr_step)

if __name__ == '__main__':
    main()