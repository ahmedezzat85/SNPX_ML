import cv2
import os
import sys
from time import time, sleep
from datetime import datetime
import argparse
from imutils.video import FPS
import utils
from multiprocessing import pool
from threading import Thread
from queue import Queue

from detector.ssd.ssd import SSDObjectDetector
from detector.yolo.yolo import YoloObjectDetector

class ObjectDetector(object):
    def __init__(self, win_name, detector, use_threading, detection_cb):
        self.detection_cb = detection_cb
        self.name  = win_name
        self.use_threading = use_threading
        self.skip_next = False
        self.stopped = False
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        self.detector = detector
        self.fps = FPS().start()
        self.t_start = datetime.now()
        if self.use_threading is True:
            self.frame_q = Queue()
            self.out_q = Queue()
            Thread(target=self.postprocess_thread).start()

    def postprocess_thread(self):
        while True:
            frame = self.frame_q.get()
            net_out = self.out_q.get()
            if frame is None: break
            if net_out is None: break
            self.bboxes = self.detector.get_bboxes(frame, net_out)
            self.detection_cb(frame, self.bboxes)
            self.frame_q.task_done()
            self.out_q.task_done()

    def __call__(self, frame):
        stop = False
        if utils.Esc_key_pressed():
            stop = True
            self.fps.stop()
            print("Elapsed = {:.2f}".format(self.fps.elapsed()))
            print("FPS     = {:.2f}".format(self.fps.fps()))

            if self.use_threading is True:
                self.out_q.put(None)
                self.frame_q.put(None)
        else:
            in_frame = self.detector.preprocess(frame)
            net_out  = self.detector.detect(in_frame)
            self.fps.update()
            if self.use_threading is True:
                self.frame_q.put(frame)
                self.out_q.put(net_out)
            else:
                self.bboxes = self.detector.get_bboxes(frame, net_out)
                self.detection_cb(frame, self.bboxes)

        return stop

class SNPXObjectDetector(object):
    """
    """
    def __init__(self, detector_type, model_name, framework):
        if detector_type.lower() == 'yolo':
            if not model_name:
                model_name = 'tiny-yolo-voc'
            self.detector = YoloObjectDetector(model=model_name, framework=framework)
        elif detector_type.lower() == 'ssd':
            if not model_name:
                model_name = 'ssd_MobileNet'
            self.detector = SSDObjectDetector(model=model_name, framework=framework)
        else:
            raise ValueError('Unknown Object Detector %s', detector_type)

    def init_cam(self, cam_id=0, cam_res=(640, 480), disp_win_title=''):
        self.cam = utils.get_default_cam(cam_id=cam_id, 
                                         win_name=disp_win_title, 
                                         resolution=cam_res)
    def set_cam(self, cam):
        self.cam = cam
    
    def start(self, use_threads=True, detection_cb=None):
        def default_cb(frame, bboxes):
            utils.draw_detections(frame, bboxes)
            cv2.imshow(self.cam.name, frame)

        callback = default_cb if detection_cb is None else detection_cb
        self.cam.start_video_capture(ObjectDetector(self.cam.name, self.detector, 
                                                    use_threads, callback))

    def stop(self):
        self.cam.close()
        self.detector.close()

def test():
    """ script entry point """
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, default='')
    parser.add_argument('-d', '--detector', type=str, default='')
    parser.add_argument('-m', '--model', type=str, default='')
    parser.add_argument('-f', '--framework', type=str, default='')
    parser.add_argument('-t', '--threading', default=False)
    args = parser.parse_args()

    cam_id = args.video if args.video else 0
    detector = SNPXObjectDetector(args.detector, args.model, args.framework)
    detector.init_cam(cam_id)
    detector.start(use_threads=args.threading)

if __name__ == '__main__':
    test()
