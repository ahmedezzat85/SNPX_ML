import os
import cv2
import numpy as np
import tensorflow as tf
import logging
import winsound

from time import time, sleep
from datetime import datetime

from darkflow.net.build import TFNet
from darkflow.defaults import argHandler
from darkflow.dark.darknet import Darknet
import utils


CURR_DIR = os.path.dirname(__file__)
BEEP_SOUND_FILE = os.path.join(CURR_DIR, 'beep.wav')

class TripWireDrawer(object):
    """ """
    def __init__(self, winname, image):
        self.done       = False
        self.drawing    = False
        self.start_pt   = (0,0)
        self.end_pt     = (0,0)
        self.color      = (0,192,0)
        self.line_width = 5
        self.frame      = image
        self.winname    = winname
 
    def draw_line(self, event, curr_pt):
        """ """
        if self.done == True:
            return
        # Start Drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = curr_pt
        # Show the line while drawing
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.line(self.frame, self.start_pt, curr_pt, self.color, self.line_width)
        # Drawing ends
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_pt  = curr_pt
            cv2.line(self.frame, self.start_pt, self.end_pt, self.color, self.line_width)

    def __call__(self):
        """ """
        def mouse_cb(event, x, y, flags, param):
            self.draw_line(event, (x,y))

        cv2.setMouseCallback(self.winname, mouse_cb)
        while True:
            cv2.imshow(self.winname, self.frame)
            if utils.is_key_pressed(utils.Enter_KEY):
                done = True
                break
        # Always set start point to lower y coordinate
        if self.end_pt[1] < self.start_pt[1]:
            tmp = self.start_pt
            self.start_pt = self.end_pt
            self.end_pt   = tmp
        return self.start_pt, self.end_pt

class TWD_Demo(object):
    """ """
    def __init__(self, cam_id=None, ip_cam=None, detector='yolo-voc'):
        if cam_id == None and ip_cam is None:
            raise ValueError('No Camera given')

        yolo_meta  = os.path.join(CURR_DIR, 'detector', detector + '.json')
        yolo_model = os.path.join(CURR_DIR, 'detector', detector + '.pb')
        yolo_args = argHandler()
        yolo_args.setDefaults()
        yolo_args['metaLoad'] = yolo_meta
        yolo_args['pbLoad']   = yolo_model
        yolo_args['gpu']      = 1.0
        yolo_args['threshold']= 0.
        self.yolo_args = yolo_args
        self.cam_name = 'Trip-Wire Detection'
        self.cam = utils.WebCam(cam_id, self.cam_name) if cam_id is not None else utils.IPCam(ip_cam, self.cam_name)

        self.tw_start = (0,0)
        self.tw_end   = (0,0)
        self.tw_slope = 0
        self.tw_color = (0, 200, 0)
        self._create_logger()

    def _create_logger(self):
        """ Create a Logger Instance."""
        # Create a new logger instance
        self.logger = logging.getLogger('TWD')
        # self.logger.setLevel(logging.INFO)
        
        ## Add a console handler
        hdlr = logging.StreamHandler()
        hdlr.setFormatter(logging.Formatter(fmt="(%(name)s)[%(levelname)s]%(message)s"))
        self.logger.addHandler(hdlr)
        self.logger.propagate = False

    def _slope(self, end_pt):
        x1, y1 = self.tw_start
        x2, y2 = end_pt
        d = (x2 - x1)
        if d == 0:
            d += 1e-8
        return (y2 - y1) / d
        
    def start_camera(self):
        """ Initialize the camera and define the trip-wire."""
        # Setup the scene of the camera
        frame = self.cam.setup_scene()

        # Draw the trip-wire
        frame = self.cam.get_frame()
        self.h, self.w, c = frame.shape
        cv2.putText(frame, 'Draw the trip-wire and press ENTER', 
                    (10, self.h//2), 3, 0.5, (0, 255, 0), 1)
        drawer = TripWireDrawer(self.cam_name, frame)
        self.tw_start, self.tw_end = drawer()
        frame = self.cam.get_frame()
        cv2.line(frame, self.tw_start, self.tw_end, self.tw_color, 2)
        cv2.imshow(self.cam_name, frame)

        # Get the line points
        self.tw_slope = self._slope(self.tw_end)
        self.h, w, c = frame.shape

    def is_event_detected(self, box, rule):
        """ """
        event_detected = False
        obj_x1, obj_x2, obj_y1, obj_y2, _, __, ___ = box
        tw_xl, tw_y1 = self.tw_start
        s11 = self._slope((obj_x1, obj_y1))
        s12 = self._slope((obj_x1, obj_y2))
        s21 = self._slope((obj_x2, obj_y1))
        s22 = self._slope((obj_x2, obj_y2))
        if rule.lower() == 'bidirectional':
            if utils.is_rect_intersected((self.tw_start, self.tw_end), 
                                         ((obj_x1, obj_y1), (obj_x2, obj_y2))) is True:
                if utils.in_frange(self.tw_slope, s11, s12) or \
                   utils.in_frange(self.tw_slope, s11, s21) or \
                   utils.in_frange(self.tw_slope, s22, s12) or \
                   utils.in_frange(self.tw_slope, s22, s21): event_detected = True

        elif rule.lower() == 'to_right':
            if (obj_x2 >= tw_xl): event_detected = True
        elif rule.lower() == 'to_left':
            if (obj_x1 <= tw_xl): event_detected = True

        self.logger.info('SLOPES  %.2f %.2f %.2f %.2f %.2f', s11, s12, s21, s22, self.tw_slope)
        self.logger.info('BOX     (%d,%d) , (%d,%d)', obj_x1, obj_y1, obj_x2, obj_y2)
        self.logger.info('Start Point  (%d,%d)', tw_xl, tw_y1)
        self.logger.info('TRUE' if event_detected is True else 'FALSE')
        return event_detected

    def run(self, rule='bidirectional', batch=1):
        """ """
        start_time = time()
        fps = 0
        self.detector = TFNet(self.yolo_args)
        colors        = self.detector.meta['colors']
        beep_on       = False
        fourcc        = cv2.VideoWriter_fourcc(*'XVID')
        vid_out_file  = os.path.join(CURR_DIR, 'twd.avi')
        writer        = cv2.VideoWriter(vid_out_file, fourcc, self.cam.fps, (self.w, self.h))
        while True:
            # Terminate if Esc is pressed
            if utils.is_key_pressed(utils.Esc_KEY): break

            # Read Frame and run the Object Detector
            frame_list = list()
            for i in range(batch):
                frame = self.cam.get_frame()
                frame = np.expand_dims(frame, 0)
                frame_list.append(frame)
            frames = np.concatenate(frame_list, 0)
            fps += batch
            box_list = self.detector.detect(frames)
            if time() - start_time >= 1:
                print ('fps = ', fps / (time() - start_time))
                fps = 0
                start_time = time()

            beep_alarm = False
            f_idx = 0
            for bboxes in box_list:
                frame = frames[f_idx]
                for box in bboxes:
                    left, right, top, bot, mess, label, confidence = box
                    rect_color = colors[label]
                    rect_thick = 1
                    if self.is_event_detected(box, rule) == True:
                        beep_alarm = True
                        rect_color = (0,0,255)
                        rect_thick = 3
                    cv2.rectangle(frame, (left, top), (right, bot), rect_color, rect_thick)
                    cv2.putText(frame, mess, (left, top - 18), 0, 1e-3 * self.h, rect_color, 2)

                # Always show the Trip-Wire line on each frame 
                cv2.line(frame, self.tw_start, self.tw_end, self.tw_color, 3)
                cv2.line(frame, self.tw_start, self.tw_end, self.tw_color, 3)
                cv2.line(frame, self.tw_start, self.tw_end, self.tw_color, 3)
                cv2.imshow(self.cam_name, frame)
                writer.write(frame)

                # Play a beep Sound if the monitored event is detected
                if beep_alarm == True:
                    beep_alarm = False
                    if beep_on == False:
                        winsound.PlaySound(BEEP_SOUND_FILE, winsound.SND_ASYNC | winsound.SND_LOOP)
                        beep_on = True
                else:
                    if beep_on == True:
                        winsound.PlaySound(None, winsound.SND_ASYNC)
                        beep_on = False

        writer.release()

# Demo
demo_vid = os.path.join(CURR_DIR, 'demo3.mp4')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.logging.set_verbosity(tf.logging.INFO)
demo = TWD_Demo(cam_id=demo_vid, detector='tiny-yolo-voc')
# demo = TWD_Demo(ip_cam='http://192.168.42.129:8080/shot.jpg', detector='tiny-yolo-voc')
demo.start_camera()
demo.run('bidirectional')
demo.cam.close()
