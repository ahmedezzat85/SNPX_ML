import numpy as np
import cv2
import utils
import argparse
from time import sleep
from imutils.video import FPS
from datetime import datetime
# from multiprocessing import Process, Queue
from threading import Thread
from queue import Queue
from detector.yolo.yolo import NCSYolo, yolo_get_candidate_objects, YoloObjectDetector
from mvnc import mvncapi as mvnc

class MvncsDetector(NCSYolo):
    """ """
    def __init__(self, model, win_name='', use_threads=True):
        super().__init__(model)
        self.graph.SetGraphOption(mvnc.GraphOption.DONT_BLOCK, 1)
        self.net_out    = None
        self.bboxes     = None
        self.in_frame   = None
        self.prev_frame = None
        self.preproc    = None
        self.started    = False
        self.win_name   = win_name
        self.use_threads = use_threads
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        if use_threads is True:
            self.inp_q   = Queue(7)
            self.stopped = False
            Thread(target=self.det_process).start()

    def det_process(self):
        while True:
            frame, preproc = self.inp_q.get()
            self.inp_q.task_done()
            if self.started is False:
                self.fps = FPS().start()
                self.started = True
            if frame is None: break
            if self.stopped is True: break
            self.process_frame(frame, preproc)
        print ('Thread Exits')

    def process_frame(self, frame, preproc):
        self.graph.LoadTensor(preproc, 'frame')

        # Process output of the previous frame
        if self.net_out is not None:
            out_shape = self.net_out.shape
            if self.model['out_size'] is not None:
                out_size = self.model['out_size']
                self.net_out = self.net_out.reshape(out_size)
                self.net_out = np.transpose(self.net_out, [2, 0, 1])
            self.net_out = self.net_out.astype(np.float32)
            self.bboxes = yolo_get_candidate_objects(self.net_out, self.prev_frame.shape)
            utils.draw_detections(self.prev_frame, self.bboxes)
            cv2.imshow(self.win_name, self.prev_frame)
            cv2.waitKey(10)
            self.fps.update()
        
        # Set the next input frame
        self.prev_frame = frame

        # Run network on the current frame
        self.net_out = None
        while self.net_out is None:
            self.net_out, obj = self.graph.GetResult()

    def call_th(self, frame):
        stop = False
        if utils.Esc_key_pressed():
            stop = True
            self.fps.stop()
            print("Elapsed = {:.2f}".format(self.fps.elapsed()))
            print("FPS     = {:.2f}".format(self.fps.fps()))
            preproc = None
            frame   = None
            self.stopped = True
        else:
            preproc = self.preprocess(frame)

        self.inp_q.put((frame, preproc))
        return stop

    def call_noth(self, frame):
        print('Time = ', datetime.now() - self.last_call)
        stop = False
        if utils.Esc_key_pressed():
            stop = True
            self.fps.stop()
            print("Elapsed = {:.2f}".format(self.fps.elapsed()))
            print("FPS     = {:.2f}".format(self.fps.fps()))
        
        # Preprocess Current frame
        if self.started is True:
            self.graph.LoadTensor(self.preproc, 'image')

        # Process output of the previous frame
        if self.net_out is not None:
            out_shape = self.net_out.shape 
            if self.model['out_size'] is not None:
                out_size = self.model['out_size']
                self.net_out = self.net_out.reshape(out_size)
                self.net_out = np.transpose(self.net_out, [2, 0, 1])
            self.net_out = self.net_out.astype(np.float32)
            self.bboxes = yolo_get_candidate_objects(self.net_out, self.in_frame.shape)
            utils.draw_detections(self.prev_frame, self.bboxes)
            cv2.imshow(self.win_name, self.prev_frame)
            self.fps.update()
        
        # Set the nest input frame
        t1 = datetime.now()
        self.prev_frame = self.in_frame
        self.in_frame   = frame
        self.preproc    = self.preprocess(self.in_frame)
        print('pre_proc = ', datetime.now() - t1)

        # Run network on the current frame
        self.net_out = None
        if self.started is True:
            while self.net_out is None:
                self.net_out, obj = self.graph.GetResult()
        else:
            self.started = True
            self.fps = FPS().start()

        self.last_call  = datetime.now()
        return stop

    def __call__(self, frame):
        return self.call_th(frame)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', default='')
    parser.add_argument('-m', '--model', default='')
    args = parser.parse_args()
    cam_id = args.video if args.video else 0
    model  = args.model 
    print ("PROFILING Model :  ", model)
    pi_cam = utils.get_default_cam(cam_id=cam_id, 
                                    win_name='Pi Camera Object Detection', 
                                    resolution=(640, 480))
    mvnc_det = MvncsDetector(model, pi_cam.name)
    pi_cam.start_video_capture(capture_cb=mvnc_det)
    mvnc_det.close()
    print ('END')

if __name__ == "__main__":
    main()
