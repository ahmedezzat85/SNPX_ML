import cv2
import os
import sys
from time import time, sleep
from datetime import datetime
import argparse
from imutils.video import FPS
import utils
from threading import Thread
from queue import Queue

from detector.ssd.ssd import SSDObjectDetector
from detector.yolo.yolo import YoloObjectDetector

class ObjectDetector(object):
    def __init__(self, win_name, detector, resolution, use_threading):
        self.name  = win_name
        self.use_threading = use_threading
        self.skip_next = False
        self.stopped = False
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        self.detector = detector
        h, w = resolution
        self.fps = FPS().start()
        self.t_start = datetime.now()
        self.in_q = Queue()
        self.frame_q = Queue()
        self.out_q = Queue()
        if self.use_threading is True:
            for task in [self.detector_thread, self.posprocess_thread]:
                Thread(target=task).start()

        # writer  = cv2.VideoWriter(out_file, fourcc, cam.fps, (w, h))

    def detector_thread(self):
        while True:
            in_frame = self.in_q.get()
            if in_frame is None: break
            # start_time = datetime.now()
            net_out = self.detector.detect(in_frame)
            # print ('Elapsed = ', str(datetime.now() - start_time))
            self.out_q.put(net_out)
            self.in_q.task_done()

    def posprocess_thread(self):
        while True:
            frame = self.frame_q.get()
            net_out = self.out_q.get()
            if frame is None: break
            if net_out is None: break
            self.bboxes = self.detector.get_bboxes(frame, net_out)
            self.frame_q.task_done()
            self.out_q.task_done()
            utils.draw_detections(frame, self.bboxes)
            cv2.imshow(self.name, frame)
            self.fps.update()

    def __call__(self, frame):
        ret = False
        if utils.Esc_key_pressed():
            ret = True
            self.fps.stop()
            print("Elapsed = {:.2f}".format(self.fps.elapsed()))
            print("FPS     = {:.2f}".format(self.fps.fps()))
            self.out_q.put(None)
            self.frame_q.put(None)
            self.in_q.put(None)
        else:
            if self.use_threading is True:
                self.frame_q.put(frame)
                frame = self.detector.preprocess(frame)
                self.in_q.put(frame)
            else:
                bboxes = self.detector(frame)
                utils.draw_detections(frame, bboxes)
                cv2.imshow(self.name, frame)
                # self.writer.write(frame)
                self.fps.update()

        return ret
        
def run_detector(cam_id, detector_type, framework, threading=True):
    # Initialize the object detector
    if detector_type.lower() == 'yolo':
        detector = YoloObjectDetector(framework=framework)
    else:
        detector = SSDObjectDetector(framework=framework)
        
    resolution = (640, 480)
    win_name   = 'Live Object Detector'
    cam = utils.get_default_cam(cam_id=cam_id, win_name=win_name, resolution=resolution,  
                            capture_cb=ObjectDetector(win_name, detector, resolution, threading))
    cam.start_video_capture()

def main():
    """ script entry point """
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, default='')
    parser.add_argument('-d', '--detector', type=str, default='yolo')
    parser.add_argument('-m', '--model', type=str, default='tiny-yolo-voc')
    parser.add_argument('-f', '--framework', type=str, default='mvncs')
    parser.add_argument('-t', '--threading', default=True)
    args = parser.parse_args()

    run_detector(args.video, args.detector, args.framework, args.threading)

if __name__ == '__main__':
    main()
