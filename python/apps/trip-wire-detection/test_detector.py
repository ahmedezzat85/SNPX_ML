import cv2
import os
import sys
from time import time
from datetime import datetime
import argparse

import utils
from imutils.video import FPS
from detector.ssd.ssd import SSDObjectDetector
from detector.yolo.yolo import YoloObjectDetector

def get_obj_detector(detector, framework):
    """ """
    if detector.lower() == 'yolo':
        det = YoloObjectDetector(framework=framework)
    else:
        det = SSDObjectDetector(framework=framework)
    return det

def detect_video(obj_det, cam_id, out_file):
    """ """
    print ('CAM = ', cam_id)
    cam = utils.WebCam(cam_id, 'caffe_webcam_demo')
    fourcc  = cv2.VideoWriter_fourcc(*'XVID')
    frame   = cam.get_frame()
    h, w, c = frame.shape
    writer  = cv2.VideoWriter(out_file, fourcc, cam.fps, (w, h))
    start_time = time()
    t_start = datetime.now()
    fps = 0
    frame_idx = 0
    while True:
        if utils.Esc_key_pressed(): break
        frame = cam.get_frame()
        if frame is None: break
        bboxes = obj_det(frame)
        utils.draw_detections(frame, bboxes)
        fps += 1
        frame_idx += 1
        if time() - start_time >= 1:
            print ('fps = ', fps / (time() - start_time))
            fps = 0
            start_time = time()
        cv2.imshow(cam.name, frame)
        writer.write(frame)
    
    print ('Elapsed = ' + str(datetime.now() - t_start))
    print ('Frame Idx = ', frame_idx)
    writer.release()
    cam.close()

def detect_images(detector, obj_det, img_list):
    """ """
    start_time = datetime.now()
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    out_dir = os.path.join(images_dir, detector)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    for img in img_list:
        image   = cv2.imread(os.path.join(images_dir, img))
        bboxes = obj_det(image)
        utils.draw_detections(image, bboxes)
        cv2.imwrite(os.path.join(out_dir, img), image)
    print ('Elapsed = ', (datetime.now() - start_time))

def main():
    """ script entry point """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--detector', type=str, default='yolo')
    parser.add_argument('-f', '--framework', type=str, default='opencv-caffe')
    parser.add_argument('-i', '--image', type=int, default=0)
    parser.add_argument('-v', '--video', type=str, default='')
    args = vars(parser.parse_args())

    framework  = args['framework']
    detector   = args['detector']
    image_test = bool(args['image'])
    det  = get_obj_detector(detector, framework)

    if image_test == True:
        images = ['office.jpg', 'dog.jpg']
        detect_images(detector, det, images)
    else:
        if args['video']:
            video_dir = os.path.join(os.path.dirname(__file__), 'video')
            out_dir   = os.path.join(video_dir, detector)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            video     = args['video']
            video_in  = os.path.join(video_dir, video)
            video_out = os.path.join(out_dir, video)
        else:
            video_in = 0
            video_out = framework + '_out.avi'
        detect_video(det, video_in, video_out)
    
if __name__ == '__main__':
    main()
