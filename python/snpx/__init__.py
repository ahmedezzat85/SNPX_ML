from __future__ import absolute_import

# Get the classifier
def get_classifier(args):
    if args.backend.lower() == 'tensorflow' or args.backend.lower() == 'tf':
        from .snpx_tf.classifier import SNPXTensorflowClassifier
        snpx_classifier = SNPXTensorflowClassifier(     model_name     = args.model_name,
                                                        dataset_name   = args.target_dataset,
                                                        data_format    = args.data_format,
                                                        use_fp16       = bool(args.use_fp16),
                                                        data_aug       = args.data_aug,
                                                        logs_root      = args.logs_dir,
                                                        model_bin_root = args.bin_dir
                                                        )
    elif args.backend.lower() == 'mxnet':
        from .snpx_mxnet.classifier import SNPXMxnetClassifier
        snpx_classifier = SNPXMxnetClassifier(          model_name     = args.model_name,
                                                        dataset_name   = args.target_dataset,
                                                        use_fp16       = bool(args.use_fp16),
                                                        data_aug       = bool(args.data_aug),
                                                        logs_root      = args.logs_dir,
                                                        model_bin_root = args.bin_dir
                                                        )
    else:
        raise ValueError('Unknown backend <%s>', args.backend)

    return snpx_classifier