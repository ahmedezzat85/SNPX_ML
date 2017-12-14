""" YOLO detection demo in Caffe """
from __future__ import print_function, division

import argparse
from time import time
from datetime import datetime
import numpy as np
import cv2

import os
os.environ["GLOG_minloglevel"] ="3"

import caffe
import utils
from yolo_post import yolo_get_candidate_objects

DETECTORS_DIR  = "D:\\SNPX_ML\\python\\apps\\trip-wire-detection\\detector"
YOLO_MODELS_DIR = os.path.join(DETECTORS_DIR, 'yolo', 'models')

class CaffeYoloDetector(object):
    """ """
    def __init__(self, yolo_model='tiny-yolo-voc', gpu=True):
        prototxt  = os.path.join(YOLO_MODELS_DIR, yolo_model + '.prototxt')
        model     = os.path.join(YOLO_MODELS_DIR, yolo_model + '.caffemodel')
        self.use_opencv = True

        if self.use_opencv == True:
            print ('OpenCV -------')
            self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        else:
            if gpu:
                print ('Caffe GPU')
                caffe.set_mode_gpu()
                caffe.set_device(0)
            else:
                print ('Caffe CPU')
                caffe.set_mode_cpu()
            self.net  = caffe.Net(prototxt, model, caffe.TEST)
            self.transformer = caffe.io.Transformer({'data': self.yolo.blobs['data'].data.shape})
            self.transformer.set_transpose('data', (2, 0, 1))
        self.mode = 'voc'

    def __call__(self, image):
        """ given a YOLO caffe model and an image, detect the objects in the image
        """
        if self.use_opencv == True:
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (416, 416)), (0.00392), (416, 416)) #0.007843
            self.net.setInput(blob)
            out = self.net.forward()[0]
        else:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.float32)
            img = img / 255.
            out = self.net.forward_all(data=np.asarray([self.transformer.preprocess('data', img)]))
            out = out['result'][0]
        bboxes = yolo_get_candidate_objects(out, image.shape, self.mode)
        return bboxes
