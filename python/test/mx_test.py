from snpx.snpx_mxnet import SNPXClassifier
import os

LOGS  = os.path.join(os.path.dirname(__file__), "..", "log")
MODEL = os.path.join(os.path.dirname(__file__), "..", "model")

classif = SNPXClassifier("mini_vgg", "CIFAR-10", devices=['GPU'],logs_root=LOGS, model_bin_root=MODEL)
classif.train(1)

