""" Synaplexus Trainer Script 
"""
import os
import snpx
from snpx_arg_parser import snpx_parse_cmd_line_options


def main():
    args = snpx_parse_cmd_line_options()
    classifier = snpx.get_classifier(args)
    classifier.evaluate(batch_size = args.batch_size)

if __name__ == '__main__':
    main()