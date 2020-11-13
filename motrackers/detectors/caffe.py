import cv2 as cv
from motrackers.detectors.detector import Detector


class Caffe_SSDMobileNet(Detector):
    def __init__(self, weights_path, configfile_path, confidence_threshold=0.5, nms_threshold=0.2,
                 draw_bboxes=True, use_gpu=False):
        object_names = {
            0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus',
            7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike',
            15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
        }

        self.pixel_mean = 127.5
        self.pixel_std = 1/127.5
        self.image_size = (300, 300)

        self.net = cv.dnn.readNetFromCaffe(configfile_path, weights_path)

        if use_gpu:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

        super().__init__(object_names, confidence_threshold, nms_threshold, draw_bboxes)

    def forward(self, image):
        blob = cv.dnn.blobFromImage(image, scalefactor=self.pixel_std, size=self.image_size,
                                    mean=(self.pixel_mean, self.pixel_mean, self.pixel_mean), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        return detections
