from collections import OrderedDict
import numpy as np
from scipy.spatial import distance
from motrackers.utils.misc import get_centroids
from motrackers.track import Track


class SimpleTracker2:
    """
    Simple Tracker.
    """
    def __init__(self, max_lost=5):
        """

        Parameters
        ----------
        max_lost : int
                   maximum number of consecutive frames object was not detected.
        """

        self.next_track_id = 0               # ID of next object
        self.tracks = OrderedDict()

        self.max_lost = max_lost
        self.frame_count = 0

    def _add_track(self, centroid, bbox, class_id, **kwargs):
        """
        Add a newly detected object to the queue

        Parameters
        ----------
        centroid : tuple
                   centroid coordinate (x, y) in pixels of the bounding box.
        bbox : tuple
               bounding box of the object being tracked as top right and bottom right coordinates.
        class_id : int
                   class label

        Returns
        -------

        """
        # store new object location
        self.tracks[self.next_track_id] = Track(track_id=self.next_track_id,
                                                centroid=centroid,
                                                bbox=bbox,
                                                class_id=class_id)
        for key, value in kwargs.items():
            self.tracks[self.next_track_id].info[key] = value

        self.next_track_id += 1

    def _remove_track(self, track_id):
        """
        Remove tracker data after object is lost

        Parameters
        ----------
        track_id : int
                    track_id of the track lost while tracking

        Returns
        -------

        """
        del self.tracks[track_id]

    def _update_track(self, track_id, centroid, bbox, **kwargs):
        self.tracks[track_id].centroid = centroid
        self.tracks[track_id].bbox = bbox
        self.tracks[track_id].lost = 0
        for key, value in kwargs.items():
            self.tracks[track_id].info[key] = value

    def _get_tracks(self, tracks):
        """
        Output the information of tracks

        Parameters
        ----------
        tracks : OrderedDict
                Dictionary of Tracks or objects being tracked with keys as track_id
                and values as corresponding `Track` objects.

        Returns
        -------
        outputs : list
                 List of tracks being currently tracked by the tracker.
                 Each element of this list contains the following tuple:
                 (frame#, trackid, class_id, centroid, bbox, info_dict).
                 class_id is the id for label of the detection.
                 centroid represents the pixel coordinates of the centroid of bounding box, i.e., (x, y).
                 bbox is the bounding box coordinates as (x_top_left, y_top_left, x_bottom_right, y_bottom_right).
                 info_dict is the dictionary of information which may be useful from the tracker (example:
                  number of times tracker was lost while tracking.).

        """
        outputs = []

        for trackid, track in tracks.items():
            track.info['lost'] = track.lost
            op = (self.frame_count, trackid, track.class_id, track.centroid, track.bbox, track.info)
            outputs.append(op)

        return outputs

    def update(self, bboxes: list, class_ids: list, detection_scores: list):
        """
        Update the tracker based on the new bboxes as input.

        Parameters
        ----------
        bboxes : list
                 List of bounding boxes detected in the current frame/timestep. Each element of the list represent
                 coordinates of bounding box as tuple (top-left-x, top-left-y, bottom-right-x, bottom-right-y).
        class_ids : list
                    List of class_ids (int) corresponding to labels of the detected object. Default is `None`.
        detection_scores: list
                         List of detection scores / probability of each detected object or objectness.

        Returns
        -------
        outputs : list
                 List of tracks being currently tracked by the tracker.
                 Each element of this list contains the following tuple:
                 (frame#, trackid, class_id, centroid, bbox, info_dict).
                 class_id is the id for label of the detection.
                 centroid represents the pixel coordinates of the centroid of bounding box, i.e., (x, y).
                 bbox is the bounding box coordinates as (x_top_left, y_top_left, x_bottom_right, y_bottom_right).
                 info_dict is the dictionary of information which may be useful from the tracker (example:
                 number of times tracker was lost while tracking.).

        """
        self.frame_count += 1

        new_bboxes = np.array(bboxes, dtype='int')
        new_class_ids = np.array(class_ids, dtype='int')
        new_detection_scores = np.array(detection_scores)

        new_centroids = get_centroids(new_bboxes)

        new_detections = list(zip(
            range(len(bboxes)), new_bboxes, new_class_ids, new_centroids, new_detection_scores
        ))

        if len(bboxes) == 0:        # if no object detected
            lost_ids = list(self.tracks.keys())
            for track_id in lost_ids:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

            outputs = self._get_tracks(self.tracks)
            return outputs

        track_ids = list(self.tracks.keys())
        if len(track_ids):
            old_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])
            D = distance.cdist(old_centroids, new_centroids)  # (row, col) = distance between old (row) and new (col)

            row_idxs = D.min(axis=1).argsort()          # old tracks sorted as per min distance from new
            col_idxs = D.argmin(axis=1)[row_idxs]       # new tracks sorted as per min distance from old

            assigned_rows, assigned_cols = set(), set()
            for (row_idx, col_idx) in zip(row_idxs, col_idxs):
                if row_idx in assigned_rows or col_idx in assigned_cols:
                    continue

                track_id = track_ids[row_idx]
                
                col_idx, bbox, class_id, centroid, detection_score = new_detections[col_idx]

                if self.tracks[track_id].class_id == class_id:
                    self._update_track(track_id, centroid, bbox, score=detection_score)
                    assigned_rows.add(row_idx)
                    assigned_cols.add(col_idx)

            unassigned_rows = set(range(0, D.shape[0])).difference(assigned_rows)
            unassigned_cols = set(range(0, D.shape[1])).difference(assigned_cols)

            if D.shape[0] >= D.shape[1]:
                for row_idx in unassigned_rows:
                    track_id = track_ids[row_idx]
                    self.tracks[track_id].lost += 1

                    if self.tracks[track_id].lost > self.max_lost:
                        self._remove_track(track_id)
            else:
                for col_idx in unassigned_cols:
                    self._add_track(new_centroids[col_idx], bboxes[col_idx], class_ids[col_idx])
        else:
            for i in range(0, len(bboxes)):
                self._add_track(new_centroids[i], bboxes[i], class_ids[i])

        outputs = self._get_tracks(self.tracks)
        return outputs
