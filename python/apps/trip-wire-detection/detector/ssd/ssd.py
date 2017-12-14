""" SSD Object Detector """
from __future__ import print_function, division

import numpy as np
import cv2
import sys
import os

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

SSD_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

class SSDObjectDetector(object):
    """ """
    def __init__(self, model='ssd_MobileNet', framework='opencv-caffe'):
        self.mean     = 127.5
        self.scale    = 0.007843
        self.in_shape = (300, 300)
        if framework.lower() == 'opencv-tf':
            prototxt  = os.path.join(SSD_MODELS_DIR, model + '.pbtxt')
            model     = os.path.join(SSD_MODELS_DIR, model + '.pb')
            self.net  = cv2.dnn.readNetFromTensorflow(model, prototxt)
            self.swapRB = True # TensorFlow uses RGB while OpenCV uses BGR
        elif framework.lower() == 'opencv-caffe':
            prototxt  = os.path.join(SSD_MODELS_DIR, model + '.prototxt')
            model     = os.path.join(SSD_MODELS_DIR, model + '.caffemodel')
            self.net  = cv2.dnn.readNetFromCaffe(prototxt, model)
            self.swapRB = False # Caffe uses BGR like OpenCV
        else:
            raise ValueError('Undefined framework %s', framework)

    def preprocess(self, image):
        return cv2.dnn.blobFromImage(cv2.resize(image, self.in_shape), self.scale, 
                                     self.in_shape, self.mean, self.swapRB)

    def detect(self, image):
        self.net.setInput(image)
        return self.net.forward()

    def get_bboxes(self, image, net_out):
        # loop over the detections
        h, w, _ = image.shape
        bboxes = list()
        for i in np.arange(0, net_out.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = net_out[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.2:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(net_out[0, 0, i, 1])
                box = net_out[0, 0, i, 3:7] * np.array([w, h, w, h])
                (xmin, ymin, xmax, ymax) = box.astype("int")

                # display the prediction
                bboxes.append([CLASSES[idx], xmin, xmax, ymin, ymax, confidence])
                
        return bboxes

    def close(self):
        pass

    def __call__(self, image):
        preprocessed = self.preprocess(image)
        net_out = self.detect(preprocessed)
        return self.get_bboxes(image, net_out)
        

def test(image_file):
    ssd = SSDObjectDetector()
    image = cv2.imread(image_file)
    ssd(image)

if __name__ == '__main__':
    test(sys.argv[1])