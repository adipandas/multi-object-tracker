import numpy as np
import cv2 as cv
from motrackers.utils.misc import xyxy2xywh


class Detector:
    """
    Abstract class for detector.

    Args:
        object_names (dict): Dictionary containing (key, value) as (class_id, class_name) for object detector.
        confidence_threshold (float): Confidence threshold for object detection.
        nms_threshold (float): Threshold for non-maximal suppression.
        draw_bboxes (bool): If true, draw bounding boxes on the image is possible.
    """

    def __init__(self, object_names, confidence_threshold, nms_threshold, draw_bboxes=True):
        self.object_names = object_names
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.height = None
        self.width = None

        np.random.seed(12345)
        if draw_bboxes:
            self.bbox_colors = {key: np.random.randint(0, 255, size=(3,)).tolist() for key in self.object_names.keys()}

    def forward(self, image):
        """
        Forward pass for the detector with input image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: detections
        """
        raise NotImplemented

    def detect(self, image):
        """
        Detect objects in the input image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            tuple: Tuple containing the following elements:
                - bboxes (numpy.ndarray): Bounding boxes with shape (n, 4) containing detected objects with each row as `(xmin, ymin, width, height)`.
                - confidences (numpy.ndarray): Confidence or detection probabilities if the detected objects with shape (n,).
                - class_ids (numpy.ndarray): Class_ids or label_ids of detected objects with shape (n, 4)

        """
        if self.width is None or self.height is None:
            (self.height, self.width) = image.shape[:2]

        detections = self.forward(image).squeeze(axis=0).squeeze(axis=0)

        bboxes, confidences, class_ids = [], [], []

        for i in range(detections.shape[0]):
            detection = detections[i, :]
            class_id = detection[1]
            confidence = detection[2]

            if confidence > self.confidence_threshold:
                bbox = detection[3:7] * np.array([self.width, self.height, self.width, self.height])
                bboxes.append(bbox.astype("int"))
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

        if len(bboxes):
            bboxes = xyxy2xywh(np.array(bboxes)).tolist()
            class_ids = np.array(class_ids).astype('int')
            indices = cv.dnn.NMSBoxes(bboxes, confidences, self.confidence_threshold, self.nms_threshold).flatten()
            return np.array(bboxes)[indices, :], np.array(confidences)[indices], class_ids[indices]
        else:
            return np.array([]), np.array([]), np.array([])

    def draw_bboxes(self, image, bboxes, confidences, class_ids):
        """
        Draw the bounding boxes about detected objects in the image.

        Args:
            image (numpy.ndarray): Image or video frame.
            bboxes (numpy.ndarray): Bounding boxes pixel coordinates as (xmin, ymin, width, height)
            confidences (numpy.ndarray): Detection confidence or detection probability.
            class_ids (numpy.ndarray): Array containing class ids (aka label ids) of each detected object.

        Returns:
            numpy.ndarray: image with the bounding boxes drawn on it.
        """

        for bb, conf, cid in zip(bboxes, confidences, class_ids):
            clr = [int(c) for c in self.bbox_colors[cid]]
            cv.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), clr, 2)
            label = "{}:{:.4f}".format(self.object_names[cid], conf)
            (label_width, label_height), baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(bb[1], label_height)
            cv.rectangle(image, (bb[0], y_label - label_height), (bb[0] + label_width, y_label + baseLine),
                         (255, 255, 255), cv.FILLED)
            cv.putText(image, label, (bb[0], y_label), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
        return image
