""" Train a classifier
"""
import threading
import numpy as np
from datetime import datetime
from time import time
import logging
import cv2
from snpx.snpx_mxnet.arch.c3d import snpx_net_create
import mxnet as mx
import mxnet.ndarray as nd
import os
from snpx.snpx_mxnet.mx_dataset import UCF101

ROOT_PATH = "D:\\PythonCode\\Base\\datasets\\UCF-101\\data\\ApplyEyeMakeup\\v_ApplyEyeMakeup_"

def vclip_2_ndarray(in_file, label, out_file):
    """ Convert a video file to mxnet.ndarray array
    """
    vframe_list = []
    cap = cv2.VideoCapture(in_file)
    while(cap.isOpened()):
        ret, vframe = cap.read()
        if not ret:
            break
        vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
        vframe = cv2.resize(vframe, (171, 128))
        vframe = nd.array(vframe)
        vframe,_ = mx.img.center_crop(vframe, (112, 112))
        vframe = nd.swapaxes(vframe, 0, 2)
        vframe = nd.expand_dims(vframe, axis=1)
        vframe_list.append(vframe)
    
    if len(vframe_list) is not 0:
        v_arr = nd.concat(*vframe_list, dim=1)
        label_arr = nd.empty((1,1))
        label_arr[0] = label
        data = {"data": v_arr.astype(np.uint8), "label": label_arr}
#        nd.save(out_file, data)
    else:
        raise ValueError("File %s Open Failed", in_file)

class PreprocessThread(threading.Thread):
    """
    """
    def __init__(self, in_file, label, out_file):
        threading.Thread.__init__(self)
        self.in_file    = in_file
        self.label      = label
        self.out_file   = out_file

    def run(self):
        vclip_2_ndarray(self.in_file, self.label, self.out_file)
        

file_list = ["g01_c01.avi", "g01_c02.avi", "g01_c03.avi", "g01_c04.avi", "g01_c05.avi", "g01_c06.avi",
             "g02_c01.avi", "g02_c02.avi", "g02_c03.avi", "g02_c04.avi",
             "g03_c01.avi", "g03_c02.avi", "g03_c03.avi", "g03_c04.avi", "g03_c05.avi", "g03_c06.avi",
             "g04_c01.avi", "g04_c02.avi", "g04_c03.avi", "g04_c04.avi"]

def profile_threading(ctx="GPU"):
    """
    """
    context = mx.cpu()
    if ctx == "GPU":
        context = mx.gpu()

    start = datetime.now()
    with mx.Context(context):
        for i in range(5):
            t = []
            for f in file_list:
                while (threading.active_count() >= 21):
                    pass
                th = PreprocessThread(ROOT_PATH+f, 1, "AAA")
                th.start()
        while (threading.active_count() > 1):
            pass
        print ("####    " + str(threading.active_count()))
    print (ctx+" Time  : " + str(datetime.now() - start))

#profile_with_threading(ctx="CPU")
#profile_threading(ctx="GPU")

u = UCF101()
u.preprocess(preprocess_threads=10)
print ("DONE!")