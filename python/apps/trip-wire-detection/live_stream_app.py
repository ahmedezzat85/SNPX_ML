import cv2
import os
import sys
from time import time
from datetime import datetime
import argparse
from imutils.video import FPS
import utils
from threading import Thread
from queue import Queue

def live_stream(cam_id):
    """ """
    # cam = utils.WebCam(cam_id, 'Live Streaming')
    cam = utils.IPCam('https://www.youtube.com/watch?v=psfFJR3vZ78', 'Live Stream')
    t_start = datetime.now()
    fps = FPS().start()
    while True:
        if utils.Esc_key_pressed(): break
        frame = cam.get_frame()
        fps.update()
        if frame is None: break
        cv2.imshow(cam.name, frame)
    
    fps.stop()
    print("Elapsed = {:.2f}".format(fps.elapsed()))
    print("FPS     = {:.2f}".format(fps.fps()))    
    cam.close()

class LiveStreamer(object):
    """
    """
    def __init__(self, cam_id):
        self.cam     = utils.WebCam(cam_id, 'Live Stream')
        self.fps     = FPS().start()
        self.in_q    = Queue()
        self.out_q   = Queue()
        self.stopped = False
        self.threads = []

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
        while True:
            if self.stopped is True: break
            frame = self.cam.get_frame()
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
            cv2.imshow(self.cam.name, frame)
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
        if utils.Esc_key_pressed():break
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
