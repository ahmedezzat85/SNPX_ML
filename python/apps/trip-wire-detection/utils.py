import os
import cv2
import numpy as np
import logging
from time import sleep

Esc_KEY   = 27
Enter_KEY = 13

def is_key_pressed(key):
    key_pressed = False
    k = cv2.waitKey(1) & 0xFF
    if k == key:
        key_pressed = True
    return key_pressed

def Esc_key_pressed():
    return is_key_pressed(Esc_KEY)

def Enter_key_pressed():
    return is_key_pressed(Enter_KEY)

def in_frange(f, float1, float2):
    """ Evaluate the expression `float1 =< f <= float2`."""
    fmin = min(float1, float2)
    fmax = max(float1, float2)
    return (fmin <= f <= fmax)

def is_rect_intersected(rect1, rect2):
    """ Check the intersection of two rectangles."""
    def _sort(a,b):
        return (a, b) if a < b else (b, a)

    (x1_a, y1_a), (x1_b, y1_b) = rect1
    (x2_a, y2_a), (x2_b, y2_b) = rect2
    x1_min, x1_max = _sort(x1_a, x1_b)
    x2_min, x2_max = _sort(x2_a, x2_b)
    y1_min, y1_max = _sort(y1_a, y1_b)
    y2_min, y2_max = _sort(y2_a, y2_b)

    if (x1_min > x2_max) or (x2_min > x1_max): return False
    if (y1_min > y2_max) or (y2_min > y1_max): return False
    return True

def draw_box(img, box, box_color=(0, 255, 0)):
    """ draw a single bounding box on the image """
    name, x_start, x_end, y_start, y_end, score = box
    h, w, _ = img.shape
    font = (1e-3 * h) * 0.5
    thick = int((h + w) // 300)
    box_tag = '{} : {: .2f}'.format(name, score)
    text_x, text_y = 5, 7

    cv2.rectangle(img, (x_start, y_start), (x_end, y_end), box_color, thick)
    boxsize, _ = cv2.getTextSize(box_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x_start, y_start-boxsize[1]-text_y),
                  (x_start+boxsize[0]+text_x, y_start), box_color, -1)
    cv2.putText(img, box_tag, (x_start+text_x, y_start-text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thick//3)

def draw_detections(img, bboxes):
    for bbox in bboxes:
        draw_box(img, bbox)


class Camera(object):
    """ Base class for Camera device.
    """
    def __init__(self, win_name, fps=8):
        self.setup = True
        self.name  = win_name
        self.fps   = fps
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
    
    def setup_scene(self):
        """ Set the fixed scene of the camera."""
        if self.setup == False: return
        while True:
            frame = self.get_frame()
            h, w, c = frame.shape
            cv2.putText(frame, 'Move the Camera to setup the scene and press Esc', 
                        (10, h//2), 3, 0.5, (0, 255, 0), 1)
            cv2.imshow(self.name, frame)
            if Esc_key_pressed(): break

class IPCam(Camera):
    """ IP Camera streaming callback.

    It returns the last captured frame for each call.

    Parameters
    ----------
    cam_url : string
        The streaming address of the camera.
    """
    def __init__(self, cam_url, win_name):
        super().__init__(win_name)
        self.url = cam_url
    
    def get_frame(self):
        cap = cv2.VideoCapture(self.url)
        _, frame = cap.read()
        cap.release()
        return frame
    
    def close(self):
        pass

try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray

    class PiWebCam(object):
        def __init__(self, cam_id=0, win_name='', resolution=(640, 480)):
            self.cam = PiCamera()
            self.name = win_name
            self.cam.resolution = resolution
            self.stopped = False
            self.setup   = True
            # self.cam.framerate = 20
 
        def setup_scene(self):
            """ Set the fixed scene of the camera."""
            def frame_cb(frame):
                stop_cap = Esc_key_pressed()
                h, w, c = frame.shape
                cv2.putText(frame, 'Move the Camera to setup the scene and press Esc', 
                            (10, h//2), 3, 0.5, (0, 255, 0), 1)
                cv2.imshow(self.name, frame)
                return stop_cap

            if self.setup == False: return
            self.start_video_capture(frame_cb)

        def start_video_capture(self, capture_cb=None):
            self.cap = PiRGBArray(self.cam, size=self.cam.resolution)
            sleep(0.1) # allow the camera to warmup
            stop_cap = False
            for pi_frame in self.cam.capture_continuous(self.cap, format="bgr", use_video_port = True):
                frame = pi_frame.array
                # Pass the frame to the callback
                if capture_cb is None:
                    cv2.imshow(self.name, frame)
                else:
                    stop_cap = capture_cb(frame)
                
                self.cap.truncate(0)
                if stop_cap is True: break
            
        def stop(self):
            self.stopped = True

        def get_frame(self):
            self.cam.capture(self.cap, format="bgr", use_video_port=True)
            frame = self.cap.array
            self.cap.truncate(0)
            return frame

        def close(self):
            """ """

except ImportError:
    class PiWebCam(object):
        def __init__(self, cam_id, win_name, resolution=(640, 480)):
            raise NotImplementedError()



class WebCam(Camera):
    """ Web Camera device capture callback.

    It returns the last captured frame for each call.

    Parameters
    ----------
    cam_id : int
        ID for the attached camera.
    """
    def __init__(self, cam_id=0, win_name='', resolution=(640, 480)):
        super().__init__(win_name)
        self.cap = cv2.VideoCapture(cam_id)
        if self.cap.isOpened() == False:
            raise ValueError('Camera not opened. Check the camera ID.')

        if isinstance(cam_id, str):
            self.setup = False
            self.fps   = round(self.cap.get(cv2.CAP_PROP_FPS))
    
    def start_video_capture(self, capture_cb=None):
        stop_cap = False
        while True:
            frame = self.get_frame()
            # Pass the frame to the callback
            if capture_cb is None:
                cv2.imshow(self.name, frame)
            else:
                stop_cap = capture_cb(frame)
            
            if stop_cap is True: break
            
    def get_frame(self):
        _, frame = self.cap.read()
        return frame

    def close(self):
        self.cap.release()

def get_default_cam(cam_type='native', **kwargs):
    if cam_type.lower() == 'ip-cam':
        default_cam = IPCam(**args)
    else:
        if 'cam_id' in kwargs:
            video = kwargs.get('cam_id')
            if video:
                return WebCam(**kwargs)

        try:
            default_cam = PiWebCam(**kwargs)
        except NotImplementedError:
            default_cam = WebCam(**kwargs)

    return default_cam

def test():
    print ('Test')
    cam = get_default_cam(cam_id=0, win_name='UTILS Test')
    while True:
        image = cam.get_frame()
        cv2.imshow(cam.name, image)
        if Esc_key_pressed(): break    
    cam.close()

if __name__ == '__main__':
    test()