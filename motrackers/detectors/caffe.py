import cv2 as cv
from motrackers.detectors.detector import Detector
from motrackers.utils.misc import load_labelsjson


class Caffe_SSDMobileNet(Detector):
    """
    Caffe SSD MobileNet model for Object Detection.

    Args:
        weights_path (str): path to network weights file.
        configfile_path (str): path to network configuration file.
        labels_path (str): path to data labels json file.
        confidence_threshold (float): confidence threshold to select the detected object.
        nms_threshold (float): Non-maximum suppression threshold.
        draw_bboxes (bool): If True, assign colors for drawing bounding boxes on the image.
        use_gpu (bool): If True, try to load the model on GPU.
    """

    def __init__(self, weights_path, configfile_path, labels_path,
                 confidence_threshold=0.5, nms_threshold=0.2, draw_bboxes=True, use_gpu=False):

        object_names = load_labelsjson(labels_path)

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
