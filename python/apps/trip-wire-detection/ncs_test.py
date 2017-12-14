import numpy as np
import cv2
import utils
from time import sleep
from imutils.video import FPS
from datetime import datetime
from detector.yolo.yolo import NCSYolo, yolo_get_candidate_objects, YoloObjectDetector
from mvnc import mvncapi as mvnc

class MvncsDetector(NCSYolo):
    """ """
    def __init__(self, model):
        super().__init__(model)
        self.graph.SetGraphOption(mvnc.GraphOption.DONT_BLOCK, 1)

    def detect_img(self, image_file, count=10):
        net_out  = None
        bboxes   = None
        in_frame = cv2.imread(image_file)
        preproc  = self.preprocess(in_frame)
        fps = FPS().start()
        t_s = datetime.now()
        while True:
            self.graph.LoadTensor(preproc, 'image')
            if net_out is not None:
                out_shape = net_out.shape 
                if self.model['out_size'] is not None:
                    out_size = self.model['out_size']
                    net_out = net_out.reshape(out_size)
                    net_out = np.transpose(net_out, [2, 0, 1])
                net_out = net_out.astype(np.float32)
                bboxes = yolo_get_candidate_objects(net_out, in_frame.shape)
                utils.draw_detections(in_frame, bboxes)
                fps.update()
                if fps._numFrames == 10:
                    fps.stop()
                    print("Elapsed = {:.2f}".format(fps.elapsed()))
                    print("FPS     = {:.2f}".format(fps.fps()))
                    break
            
            in_frame = cv2.imread(image_file)
            preproc  = self.preprocess(in_frame)
            net_out = None
            while net_out is None:
                net_out, obj = self.graph.GetResult()
        print ('Time = ', datetime.now() - t_s)

def test_non_blocking():
    image_file = 'images/office.jpg'
    for model in ['tiny-yolo-v1', 'tiny-yolo-v2']:
        print ("PROFILING Model :  ", model)
        mvnc_det = MvncsDetector(model)
        mvnc_det.detect_img(image_file)
        mvnc_det.close()
        sleep(3)

    
def main():
    image_file = 'images/office.jpg'
    for model in ['tiny-yolo-v1', 'tiny-yolo-v2']:
        print ("PROFILING Model :  ", model)
        mvnc_det = YoloObjectDetector(model, 'mvncs')

        fps = FPS().start()
        image = cv2.imread(image_file)
        pre_proc = mvnc_det.preprocess(image)
        for i in range(10):
            t_s = datetime.now()
            net_out = mvnc_det.detect(pre_proc)
            print ('Time = ', datetime.now() - t_s)
            bboxes = mvnc_det.get_bboxes(image, net_out)
            utils.draw_detections(image, bboxes)
            fps.update()
        fps.stop()
        
        print("Elapsed = {:.2f}".format(fps.elapsed()))
        print("FPS     = {:.2f}".format(fps.fps()))
        cv2.imwrite(model+'.jpg', image)
        mvnc_det.close()
        sleep(2)

if __name__ == "__main__":
    test_non_blocking()
    # main()
