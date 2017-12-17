# import os
# from snpx.snpx_tf import SNPXClassifier
# from snpx.snpx_tf.tf_dataset import CIFAR10
# import tensorflow as tf
# import numpy as np

# data_format = "NHWC"
# batch_size  = 256

# LOGS  = os.path.join(os.path.dirname(__file__), "..", "log")
# MODEL = os.path.join(os.path.dirname(__file__), "..", "model")

# classif = SNPXClassifier(model_name     = "mini_vgg_bn", 
#                          dataset        = "CIFAR-10",
#                          devices        = ['GPU'], 
#                          logs_root      = LOGS, 
#                          model_bin_root = MODEL,
#                          data_format    = data_format,
#                          use_fp16       = False,
#                          debug          = False)
# classif.train(1, batch_size=batch_size)


# classif.deploy_model(image_size=[32, 32])

def fn(a,b):
    print (a,b)

arg = {'a': 1, 'b': 2}
fn(**arg)