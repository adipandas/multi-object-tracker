"""
Implementation of this algorithm is heavily based on the following:
https://github.com/bochinski/iou-tracker
"""

import numpy as np
from motrackers.utils import get_centroids, iou
from motrackers import SimpleTracker2


class IOUTracker(SimpleTracker2):
    def __init__(self, max_lost=2, iou_threshold=0.5, min_detection_confidence=0.4, max_detection_confidence=0.7):
        self.iou_threshold = iou_threshold
        self.max_detection_confidence = max_detection_confidence
        self.min_detection_confidence = min_detection_confidence
        super(IOUTracker, self).__init__(max_lost=max_lost)

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
                 Each element of this list contains the tuple in
                 format (frame#, trackid, class_id, centroid, bbox, info_dict).
                 class_id is the id for label of the detection.
                 centroid represents the pixel coordinates of the centroid of bounding box, i.e., (x, y).
                 bbox is the bounding box coordinates as (x_top_left, y_top_left, x_bottom_right, y_bottom_right).
                 info_dict is the dictionary of information which may be useful from the tracker (example:
                 number of times tracker was lost while tracking.).

        """

        assert len(bboxes) == len(class_ids), "Containers must be of same length. len(bboxes)={}," \
                                              " len(class_ids)={}".format(len(bboxes), len(class_ids))

        assert len(bboxes) == len(detection_scores), "Containers must be of same length. len(bboxes)={}," \
                                                     " len(class_ids)={}".format(len(bboxes), len(detection_scores))

        self.frame_count += 1

        new_bboxes = np.array(bboxes, dtype='int')
        new_class_ids = np.array(class_ids, dtype='int')
        new_detection_scores = np.array(detection_scores)

        new_centroids = get_centroids(new_bboxes)

        new_detections = list(zip(
            range(len(bboxes)), new_bboxes, new_class_ids, new_centroids, new_detection_scores
        ))

        track_ids = list(self.tracks.keys())

        updated_tracks = []
        for track_id in track_ids:
            if len(new_detections) > 0:
                idx, bb, cid, ctrd, scr = max(new_detections, key=lambda x: iou(self.tracks[track_id].bbox, x[1]))

                if iou(self.tracks[track_id].bbox, bb) > self.iou_threshold and self.tracks[track_id].class_id == cid:
                    max_score = max(self.tracks[track_id].info['max_score'], scr)
                    self._update_track(track_id, ctrd, bb, score=scr, max_score=max_score)

                    updated_tracks.append(track_id)

                    del new_detections[idx]

            if len(updated_tracks) == 0 or track_id is not updated_tracks[-1]:
                self.tracks[track_id].lost += 1

                if self.tracks[track_id].lost > self.max_lost and \
                        self.tracks[track_id].info['max_score'] >= self.max_detection_confidence:
                    self._remove_track(track_id)

        for idx, bb, cid, ctrd, scr in new_detections:
            self._add_track(ctrd, bb, cid, score=scr, max_score=scr)

        outputs = self._get_tracks(self.tracks)
        return outputs
