import numpy as np
from scipy.optimize import linear_sum_assignment
from motrackers.utils.misc import iou
from motrackers.kalman_tracker import KFTrackerSORT
from motrackers.tracker import CentroidKF_Tracker


def associate_tracks_to_detection(bbox_tracks, bbox_detections, iou_threshold=0.3):
    """
    Assigns detected bounding boxes to tracked bounding boxes using IoU as a distance metric.

    Parameters
    ----------
    bbox_tracks : numpy.ndarray
    bbox_detections : numpy.ndarray
    iou_threshold : float

    Returns
    -------
    tuple :
        Tuple containing the following elements
            - matches: (numpy.ndarray) Array of shape `(n, 2)` where `n` is number of pairs formed after
                matching tracks to detections. This is an array of tuples with each element as matched pair
                of indices`(track_index, detection_index)`.
            - unmatched_detections : (numpy.ndarray) Array of shape `(m,)` where `m` is number of unmatched detections.
            - unmatched_tracks : (numpy.ndarray) Array of shape `(k,)` where `k` is the number of unmatched tracks.
    """

    if (bbox_tracks.size == 0) or (bbox_detections.size == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(bbox_detections), dtype=int), np.empty((0,), dtype=int)

    if len(bbox_tracks.shape) == 1:
        bbox_tracks = bbox_tracks[None, :]

    if len(bbox_detections.shape) == 1:
        bbox_detections = bbox_detections[None, :]

    iou_matrix = np.zeros((bbox_tracks.shape[0], bbox_detections.shape[0]), dtype=np.float32)

    for t in range(bbox_tracks.shape[0]):
        for d in enumerate(bbox_detections.shape[0]):
            iou_matrix[t, d] = iou(bbox_tracks[t, :], bbox_detections[d, :])

    assigned_tracks, assigned_detections = linear_sum_assignment(-iou_matrix)

    unmatched_detections, unmatched_tracks = [], []

    for d in range(bbox_detections.shape[0]):
        if d not in assigned_detections:
            unmatched_detections.append(d)

    for t in range(bbox_tracks.shape[0]):
        if t not in assigned_tracks:
            unmatched_tracks.append(t)

    # filter out matched with low IOU
    matches = []
    for t, d in zip(assigned_tracks, assigned_detections):
        if iou_matrix[t, d] < iou_threshold:
            unmatched_detections.append(d)
            unmatched_tracks.append(t)
        else:
            matches.append((t, d))

    if len(matches):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)


class SORT(CentroidKF_Tracker):
    """
    This MOT Tracker is based on SORT.

    Parameters
    ----------
    max_lost : int
        Max. number of times a object is lost while tracking.
    min_hits : int
        Minimum length of time an object is tracked.
    tracker_output_format : str
        Output format of the tracker.
    iou_threshold : float
        Intersection over union minimum value.
    process_noise_covariance : float or numpy.ndarray
        Process noise covariance matrix of shape (3, 3) or covariance magnitude as scalar value.
    measurement_noise_covariance : float or numpy.ndarray
        Measurement noise covariance matrix of shape (1,) or covariance magnitude as scalar value.
    time_step : int or float
        Time step for Kalman Filter.
    """

    def __init__(
            self, max_lost=1, min_hits=3,
            tracker_output_format='mot_challenge',
            iou_threshold=0.3,
            process_noise_covariance=None,
            measurement_noise_covariance=None,
            time_step=1
    ):

        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        super().__init__(
            max_lost=max_lost, tracker_output_format=tracker_output_format,
            process_noise_covariance=process_noise_covariance,
            measurement_noise_covariance=measurement_noise_covariance, time_step=time_step
        )

    @staticmethod
    def measurement_from_bbox(bbox):
        """
        Convert bounding box tuple from (xmin, ymin, width, height) to (x_centroid, y_centroid, width, aspect_ratio)
        Parameters
        ----------
        bbox : numpy.ndarray or list

        Returns
        -------
        bbox_measurement : numpy.ndarray or list
        """

        x, y, w, h = bbox
        x_centroid, y_centroid = x + 0.5 * w, y + 0.5 * h
        aspect_ratio = w/float(h)

        return np.array([x_centroid, y_centroid, w, aspect_ratio])

    @staticmethod
    def bbox_from_measurement(measurement):
        """
        Convert bounding box coordinates from tuple of format `(x_centroid, y_centroid, width, aspect_ratio)`
        to `(xmin, ymin, width, height)`.

        Parameters
        ----------
        measurement : numpy.ndarray
            Bounding box coordinates from Kalman filter prediction as `(x_centroid, y_centroid, width, aspect_ratio)`.

        Returns
        -------
        bbox : numpy.ndarray
            Estimated bounding box coordinates as `(xmin, ymin, width, height)`.

        """

        xc, yc, w, ar = measurement
        x = xc - w*0.5
        h = w/float(ar)
        y = yc - 0.5*h
        return np.array([x, y, w, h])

    def _add_kf_tracker(self, track_id, bbox):
        m = self.measurement_from_bbox(bbox)
        initial_state = np.array([m[0], 0, 0, m[1], 0, 0, m[2], 0, 0, m[3]])
        kf = KFTrackerSORT(time_step=self.time_step)
        kf.setup(
            process_noise_covariance=self.process_noise_covariance,
            measurement_noise_covariance=self.measurement_noise_covariance,
            initial_state=initial_state
        )
        self.kalman_trackers[track_id] = kf

    def _predict_kf_tracker(self, track_id):
        prediction = self.kalman_trackers[track_id].predict()
        bbox = self.bbox_from_measurement(prediction).astype('int')
        return bbox
    
    def _update_kf_tracker(self, track_id, bbox):
        measurement = self.measurement_from_bbox(bbox)
        self.kalman_trackers[track_id].update(measurement)

    def update(self, bboxes, detection_scores, class_ids):
        self.frame_count += 1

        track_ids = list(self.tracks.keys())

        bbox_detections = np.array(bboxes, dtype='int')

        bbox_tracks = []
        for track_id in track_ids:
            bbox_tracks.append(self._predict_kf_tracker(track_id))

        bbox_tracks = np.array(bbox_tracks)

        matches, unmatched_detections, unmatched_tracks = associate_tracks_to_detection(
            bbox_tracks, bbox_detections, iou_threshold=0.3
        )

        for i in range(matches.shape[0]):
            t, d = matches[i, :]
            track_id = track_ids[t]
            bbox = bboxes[d, :]
            cid = class_ids[d]
            confidence = detection_scores[d]
            self._update_track(track_id, self.frame_count, bbox, confidence, cid, lost=0)
            self._update_kf_tracker(track_id, bbox)
            
        for d in unmatched_detections:
            bbox = bboxes[d, :]
            cid = class_ids[d]
            confidence = detection_scores[d]
            self._add_track(self.frame_count, bbox, confidence, cid)
            self._add_kf_tracker(self.next_track_id-1, bbox)

        for t in unmatched_tracks:
            track_id = track_ids[t]
            bbox = bbox_tracks[track_id, :]
            confidence = self.tracks[track_id].detection_confidence
            cid = self.tracks[track_id].class_id
            self._update_track(track_id, self.frame_count, bbox, detection_confidence=confidence, class_id=cid, lost=1)
            
            if self.tracks[track_id].lost > self.max_lost:
                self._remove_track(track_id)
                self._remove_kf_tracker(track_id)

        outputs = self._get_tracks(self.tracks)
        return outputs
