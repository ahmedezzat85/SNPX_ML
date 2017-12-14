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

from picamera import PiCamera
from picamera.array import PiRGBArray

def init_pi_cam(win_name, resolution=(640, 480)):
    """ Web Camera device capture callback.

    It returns the last captured frame for each call.

    Parameters
    ----------
    cam_id : int
        ID for the attached camera.
    """
    cam = PiCamera()
    cam.resolution = resolution
    cam.framerate = 20
    cap = PiRGBArray(cam, size=resolution)
    # allow the camera to warmup
    sleep(0.25)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    return cam, cap

def live_stream(cam_id):
    """ """
    win_name = 'Pi Live Streaming'
    cam, cap = init_pi_cam(win_name)
    t_start = datetime.now()
    fps = FPS().start()
    for pi_frame in cam.capture_continuous(cap, format="bgr", use_video_port = True):
        frame = pi_frame.array
        fps.update()
        cv2.imshow(win_name, frame)
        cap.truncate(0)
        if utils.Esc_key_pressed(): break
    
    fps.stop()
    print("Elapsed = {:.2f}".format(fps.elapsed()))
    print("FPS     = {:.2f}".format(fps.fps()))    

class LiveStreamer(object):
    """
    """
    def __init__(self, cam_id):
        self.win_name = 'Pi Live Streaming'
        self.cam, self.cap = init_pi_cam(self.win_name)
        self.fps      = FPS().start()
        self.in_q     = Queue()
        self.out_q    = Queue()
        self.stopped  = False
        self.threads  = []

    def start(self):
         th = Thread(target=self.frame_preprocess)
         th.start()
         self.threads.append(th)

         th = Thread(target=self.frame_process)
         th.start()
         self.threads.append(th)

         th = Thread(target=self.stream)
         th.start()
         self.threads.append(th)

    def frame_preprocess(self):
        for pi_frame in self.cam.capture_continuous(self.cap, format="bgr", use_video_port = True):
            if self.stopped is True: break
            frame = pi_frame.array
            self.in_q.put(frame)
            if frame is None: break
        self.stopped = True

    def frame_process(self):
        while True:
            if self.stopped is True: break
            frame = self.in_q.get()
            if frame is None: break
            self.fps.update()
            self.out_q.put(frame)
            self.in_q.task_done()
        self.stopped = True

    def stream(self):
        while True:
            frame = self.out_q.get()
            if frame is None: break
            cv2.imshow(self.win_name, frame)
            self.out_q.task_done()
        self.stopped = True

    def stop(self):
        self.stopped = True
        self.in_q.put(None)
        self.out_q.put(None)
        for th in self.threads:
            th.join()

        self.fps.stop()
        print("Elapsed    = {:.2f} sec".format(self.fps.elapsed()))
        print("Frame Rate = {:.2f} fps".format(self.fps.fps()))    
        self.cam.close()

def start_live_stream(cam_id):
    """ """
    streamer = LiveStreamer(cam_id)
    streamer.start()
    while True:
        if utils.Esc_key_pressed(): break
        if streamer.stopped == True: break
    streamer.stop()

def main():
    """ script entry point """
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, default='')
    args = vars(parser.parse_args())
    if args['video']:
        video     = args['video']
    else:
        video = 0
    live_stream(video)
    # start_live_stream(video)
    
if __name__ == '__main__':
    main()
