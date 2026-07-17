import numpy as np
import pytest

from motrackers import CentroidTracker, IOUTracker, CentroidKF_Tracker, SORT
from motrackers.track import Track


ALL_TRACKERS = [CentroidTracker, IOUTracker, CentroidKF_Tracker, SORT]


def _detections():
    bboxes = np.array([[10, 10, 20, 20], [200, 200, 30, 30]], dtype=int)
    scores = np.array([0.9, 0.8])
    class_ids = np.array([1, 2])
    return bboxes, scores, class_ids


@pytest.mark.parametrize("tracker_cls", ALL_TRACKERS)
def test_tracker_creates_tracks(tracker_cls):
    tracker = tracker_cls()
    bboxes, scores, class_ids = _detections()
    outputs = tracker.update(bboxes, scores, class_ids)
    assert len(outputs) == 2
    for track in outputs:
        assert len(track) == 10


@pytest.mark.parametrize("tracker_cls", ALL_TRACKERS)
def test_tracker_ids_are_integers_and_unique(tracker_cls):
    tracker = tracker_cls()
    bboxes, scores, class_ids = _detections()
    outputs = tracker.update(bboxes, scores, class_ids)
    ids = [int(track[1]) for track in outputs]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("tracker_cls", ALL_TRACKERS)
def test_tracker_handles_empty_detections(tracker_cls):
    tracker = tracker_cls()
    empty = np.empty((0, 4), dtype=int)
    outputs = tracker.update(empty, np.array([]), np.array([], dtype=int))
    assert outputs == []


def test_centroid_tracker_keeps_id_across_frames():
    tracker = CentroidTracker()
    bboxes = np.array([[10, 10, 20, 20]], dtype=int)
    scores = np.array([0.9])
    class_ids = np.array([1])

    first = tracker.update(bboxes, scores, class_ids)
    # move the object slightly; it should remain the same track id
    bboxes_moved = np.array([[12, 12, 20, 20]], dtype=int)
    second = tracker.update(bboxes_moved, scores, class_ids)

    assert int(first[0][1]) == int(second[0][1])


def test_centroid_tracker_removes_lost_track():
    tracker = CentroidTracker(max_lost=1)
    bboxes = np.array([[10, 10, 20, 20]], dtype=int)
    tracker.update(bboxes, np.array([0.9]), np.array([1]))

    empty = np.empty((0, 4), dtype=int)
    tracker.update(empty, np.array([]), np.array([], dtype=int))  # lost = 1
    outputs = tracker.update(empty, np.array([]), np.array([], dtype=int))  # lost > max_lost
    assert outputs == []
