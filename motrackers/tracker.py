from collections import OrderedDict
import numpy as np
from scipy.spatial import distance
from motrackers.utils.misc import get_centroid
from motrackers.track import Track
from motrackers.kalman_tracker import KFTracker2D
from scipy.optimize import linear_sum_assignment


class Tracker:
    """
    Greedy Tracker with tracking based on `centroid` location of the bounding box of the object.

    References
    ----------
    This tracker is also referred as `CentroidTracker` in this repository.

    Parameters
    ----------
    max_lost : int
        Maximum number of consecutive frames object was not detected.
    tracker_output_format : str
        Output format of the tracker.
    """

    def __init__(self, max_lost=5, tracker_output_format='mot_challenge'):
        self.next_track_id = 0
        self.tracks = OrderedDict()
        self.max_lost = max_lost
        self.frame_count = 0
        self.tracker_output_format = tracker_output_format

    def _add_track(self, frame_id, bbox, detection_confidence, class_id, **kwargs):
        """
        Add a newly detected object to the queue.

        Parameters
        ----------
        frame_id : int
            Camera frame id.
        bbox : numpy.ndarray
            Bounding box pixel coordinates as (xmin, ymin, xmax, ymax) of the track.
        detection_confidence : float
            Detection confidence of the object (probability).
        class_id : Object
            Class label id.
        kwargs : dict
            Additional key word arguments.
        """
        self.tracks[self.next_track_id] = Track(
            self.next_track_id, frame_id, bbox, detection_confidence, class_id=class_id,
            data_output_format=self.tracker_output_format,
            **kwargs
        )
        self.next_track_id += 1

    def _remove_track(self, track_id):
        """
        Remove tracker data after object is lost.

        Parameters
        ----------
        track_id : int
                    track_id of the track lost while tracking
        """
        del self.tracks[track_id]

    def _update_track(self, track_id, frame_id, bbox, detection_confidence, class_id, lost=0, iou_score=0., **kwargs):
        """
        Update track state.

        Parameters
        ----------
        track_id : int
            ID of the track.
        frame_id : int
            Frame count.
        bbox : numpy.ndarray or list
            Bounding box coordinates as (xmin, ymin, width, height)
        detection_confidence : float
            Detection confidence (aka detection probability).
        class_id : int
            ID of the class (aka label) of the object being tracked.
        lost : int
            Number of frames the object was lost while tracking.
        iou_score : float
            Intersection over union.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------

        """

        self.tracks[track_id].update(
            frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs
        )

    @staticmethod
    def _get_tracks(tracks):
        """
        Output the information of tracks.

        Parameters
        ----------
        tracks : OrderedDict
            Tracks dictionary with (key, value) as (track_id, corresponding `Track` objects).

        Returns
        -------
        outputs : list
            List of tracks being currently tracked by the tracker.
        """

        outputs = []
        for trackid, track in tracks.items():
            outputs.append(track.output())
        return outputs

    @staticmethod
    def preprocess_input(bboxes, class_ids, detection_scores):
        """
        Preprocess the input data.

        Parameters
        ----------
        bboxes : list or numpy.ndarray
            Array of bounding boxes with each bbox as a tuple containing `(xmin, ymin, width, height)`.
        class_ids : list or numpy.ndarray
            Array of Class ID or label ID.
        detection_scores : list or numpy.ndarray
            Array of detection scores (aka. detection probabilities).

        Returns
        -------
        detections : list[Tuple]
            Data for detections as list of tuples containing `(bbox, class_id, detection_score)`.
        """
        new_bboxes = np.array(bboxes, dtype='int')
        new_class_ids = np.array(class_ids, dtype='int')
        new_detection_scores = np.array(detection_scores)

        new_detections = list(zip(new_bboxes, new_class_ids, new_detection_scores))
        return new_detections

    def update(self, bboxes, detection_scores, class_ids):
        """
        Update the tracker based on the new bounding boxes.

        Parameters
        ----------
        bboxes : numpy.ndarray or list
            List of bounding boxes detected in the current frame. Each element of the list represent
            coordinates of bounding box as tuple `(top-left-x, top-left-y, bottom-right-x, bottom-right-y)`.
        detection_scores: numpy.ndarray or list
            List of detection scores (probability) of each detected object.
        class_ids : numpy.ndarray or list
            List of class_ids (int) corresponding to labels of the detected object. Default is `None`.

        Returns
        -------
        outputs : list
            List of tracks being currently tracked by the tracker. Each track is represented by the tuple with elements
            `(frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)`.
        """

        self.frame_count += 1

        if len(bboxes) == 0:
            lost_ids = list(self.tracks.keys())

            for track_id in lost_ids:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

            outputs = self._get_tracks(self.tracks)
            return outputs

        detections = Tracker.preprocess_input(bboxes, class_ids, detection_scores)

        track_ids = list(self.tracks.keys())

        updated_tracks, updated_detections = [], []

        if len(track_ids):
            track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])
            detection_centroids = get_centroid(bboxes)

            centroid_distances = distance.cdist(track_centroids, detection_centroids)

            track_indices = np.amin(centroid_distances, axis=1).argsort()

            for idx in track_indices:
                track_id = track_ids[idx]

                remaining_detections = [
                    (i, d) for (i, d) in enumerate(centroid_distances[idx, :]) if i not in updated_detections]

                if len(remaining_detections):
                    detection_idx, detection_distance = min(remaining_detections, key=lambda x: x[1])
                    bbox, class_id, confidence = detections[detection_idx]
                    self._update_track(track_id, self.frame_count, bbox, confidence, class_id=class_id)
                    updated_detections.append(detection_idx)
                    updated_tracks.append(track_id)

                if len(updated_tracks) == 0 or track_id is not updated_tracks[-1]:
                    self.tracks[track_id].lost += 1
                    if self.tracks[track_id].lost > self.max_lost:
                        self._remove_track(track_id)

        for i, (bbox, class_id, confidence) in enumerate(detections):
            if i not in updated_detections:
                self._add_track(self.frame_count, bbox, confidence, class_id=class_id)

        outputs = self._get_tracks(self.tracks)
        return outputs


def assign_tracks2detection(bbox_tracks, bbox_detections, distance_threshold=0.5):
    """
    Assigns detected bounding boxes to tracked bounding boxes using IoU as a distance metric.

    Parameters
    ----------
    bbox_tracks : numpy.ndarray
    bbox_detections : numpy.ndarray
    distance_threshold : float

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

    estimated_track_centroids = get_centroid(bbox_tracks)
    detection_centroids = get_centroid(bbox_detections)
    centroid_distances = distance.cdist(estimated_track_centroids, detection_centroids)

    assigned_tracks, assigned_detections = linear_sum_assignment(centroid_distances)

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
        if centroid_distances[t, d] < distance_threshold:
            unmatched_detections.append(d)
            unmatched_tracks.append(t)
        else:
            matches.append((t, d))

    if len(matches):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)


class CentroidKF_Tracker(Tracker):
    """
    Kalman filter based tracking of multiple detected objects.

    Parameters
    ----------
    max_lost : int
        Maximum number of consecutive frames object was not detected.
    tracker_output_format : str
        Output format of the tracker.
    process_noise_covariance : float or numpy.ndarray
        Process noise covariance matrix of shape (3, 3) or covariance magnitude as scalar value.
    measurement_noise_covariance : float or numpy.ndarray
        Measurement noise covariance matrix of shape (1,) or covariance magnitude as scalar value.
    time_step : int or float
        Time step for Kalman Filter.
    """
    def __init__(
            self,
            max_lost=5,
            tracker_output_format='mot_challenge',
            process_noise_covariance=None,
            measurement_noise_covariance=None,
            time_step=1
    ):
        self.time_step = time_step
        self.process_noise_covariance = process_noise_covariance
        self.measurement_noise_covariance = measurement_noise_covariance

        self.kalman_trackers = OrderedDict()
        super().__init__(max_lost, tracker_output_format)

    def _add_kf_tracker(self, track_id, track_centroid):
        """
        Add kalman tracker for the object to be tracked.

        Parameters
        ----------
        track_id : int
            Tracker ID (aka object ID) to be tracked.
        track_centroid : numpy.ndarray or list
            Pixel coordinates of centroid of the track as (centroid_x, centroid_y) as initial state of
            the object being tracked using the Kalman Filter.

        Returns
        -------
        """

        kf = KFTracker2D(time_step=self.time_step)
        kf.setup(
            process_noise_covariance_x=self.process_noise_covariance,
            measurement_noise_covariance_x=self.measurement_noise_covariance,
            initial_state_x=track_centroid[0],
            process_noise_covariance_y=self.process_noise_covariance,
            measurement_noise_covariance_y=self.measurement_noise_covariance,
            initial_state_y=track_centroid[1]
        )

        self.kalman_trackers[track_id] = kf

    def _predict_kf_tracker(self, track_id):
        """
        Use Kalman Tracker to predict the current location of the object being tracked.

        Parameters
        ----------
        track_id : int
            ID of the object to predict the location for.

        Returns
        -------
        bbox : numpy.ndarray
            Bounding box of the object as `(xmin, ymin, width, height)`.

        """
        centroid_x, centroid_y = self.kalman_trackers[track_id].predict()
        w, h = self.tracks[track_id].bbox[2], self.tracks[track_id].bbox[3]
        x, y = int(centroid_x - 0.5 * w), int(centroid_y - 0.5 * h)
        return np.array([x, y, w, h])

    def _remove_kf_tracker(self, track_id):
        """
        Remove KF tracker data after object is lost.

        Parameters
        ----------
        track_id : int
            track_id of the track lost while tracking
        """
        del self.kalman_trackers[track_id]

    def _update_kf_tracker(self, track_id, track_centroid):
        """
        Update Kalman Tracker of the object being tracked with the new measurement.

        Parameters
        ----------
        track_id : int
            ID of the object.
        track_centroid : numpy.ndarray
            Centroid pixel coordinates with shape (2,) of the object being tracked, i.e., `(x, y)`.

        """
        self.kalman_trackers[track_id].update(track_centroid)

    def update(self, bboxes, detection_scores, class_ids):
        self.frame_count += 1

        if len(bboxes) == 0:
            lost_ids = list(self.tracks.keys())
            for track_id in lost_ids:
                estimated_bbox = self._predict_kf_tracker(track_id)

                self._update_track(track_id, self.frame_count, estimated_bbox, detection_confidence=0.,
                                   class_id=self.tracks[track_id].class_id, lost=1)

                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)
                    self._remove_kf_tracker(track_id)
                    
            outputs = self._get_tracks(self.tracks)
            return outputs

        detections = Tracker.preprocess_input(bboxes, class_ids, detection_scores)

        track_ids = list(self.tracks.keys())
        n_tracks, n_detections = len(track_ids), len(detections)

        if len(track_ids):
            estimated_track_bboxes = np.array([self._predict_kf_tracker(tid) for tid in track_ids])
            estimated_track_centroids = get_centroid(estimated_track_bboxes)
            detection_centroids = get_centroid(bboxes)
            centroid_distances = distance.cdist(estimated_track_centroids, detection_centroids)
            track_indices, detection_indices = linear_sum_assignment(centroid_distances)

            for (tidx, didx) in zip(track_indices, detection_indices):
                track_id = track_ids[tidx]
                bbox, class_id, confidence = detections[didx]
                self._update_track(track_id, self.frame_count, bbox, confidence, class_id=class_id)
                self._update_kf_tracker(track_id, self.tracks[track_id].centroid)

            if n_tracks > n_detections:
                mask = np.ones((n_tracks,), dtype=bool)
                mask[track_indices] = False
                unassigned_track_indices = track_indices[mask]

                for tidx in unassigned_track_indices:
                    track_id = track_ids[tidx]
                    self._update_track(track_id, self.frame_count, estimated_track_bboxes[track_id],
                                       detection_confidence=0., class_id=self.tracks[track_id].class_id, lost=1)
                    if self.tracks[track_id].lost > self.max_lost:
                        self._remove_track(track_id)
                        self._remove_kf_tracker(track_id)

            elif n_tracks < n_detections:
                mask = np.ones((n_detections,), dtype=bool)
                mask[detection_indices] = False
                unassigned_detection_indices = np.arange(n_detections)[mask]
                for didx in unassigned_detection_indices:
                    bbox, class_id, confidence = detections[didx]
                    self._add_track(self.frame_count, bbox, confidence, class_id=class_id)
                    self._add_kf_tracker(self.next_track_id-1, self.tracks[self.next_track_id-1].centroid)

        else:
            for detection in detections:
                bbox, class_id, confidence = detection
                self._add_track(self.frame_count, bbox, confidence, class_id=class_id)
                self._add_kf_tracker(self.next_track_id - 1, self.tracks[self.next_track_id - 1].centroid)

        outputs = self._get_tracks(self.tracks)
        return outputs
