import numpy as np
import pytest

from motrackers.track import Track, KFTrackCentroid


def _make_base_track():
    return Track(
        track_id=0,
        frame_id=1,
        bbox=np.array([10, 20, 30, 40]),
        detection_confidence=0.9,
        class_id=1,
    )


def test_base_track_predict_raises_not_implemented():
    track = _make_base_track()
    with pytest.raises(NotImplementedError):
        track.predict()


def test_mot_challenge_output_format():
    track = _make_base_track()
    out = track.get_mot_challenge_format()
    assert len(out) == 10
    frame_id, track_id, left, top, w, h, conf, x, y, z = out
    assert (frame_id, track_id) == (1, 0)
    assert (left, top, w, h) == (10, 20, 30, 40)
    assert conf == pytest.approx(0.9)
    assert (x, y, z) == (-1, -1, -1)


def test_visdrone_output_format():
    track = Track(
        track_id=3,
        frame_id=2,
        bbox=np.array([1, 2, 3, 4]),
        detection_confidence=0.5,
        class_id=7,
        data_output_format='visdrone_challenge',
    )
    out = track.get_vis_drone_format()
    assert len(out) == 10
    assert out[7] == 7  # object_category = class_id


def test_invalid_output_format_raises():
    with pytest.raises(AssertionError):
        Track(0, 1, np.array([0, 0, 1, 1]), 0.5, class_id=1, data_output_format='unknown')


def test_centroid_kf_track_predict_returns_bbox():
    track = KFTrackCentroid(
        track_id=0,
        frame_id=1,
        bbox=np.array([10, 20, 30, 40]),
        detection_confidence=0.9,
        class_id=1,
    )
    bb = track.predict()
    assert bb.shape == (4,)
