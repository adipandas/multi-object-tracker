import numpy as np
import pytest

from motrackers.utils.misc import (
    get_centroid,
    iou,
    iou_xywh,
    xyxy2xywh,
    xywh2xyxy,
    midwh2xywh,
    intersection_complement_indices,
    nms,
)


def test_get_centroid_single():
    bbox = np.array([10, 20, 40, 60])  # xmin, ymin, w, h
    c = get_centroid(bbox)
    assert c.shape == (2,)
    np.testing.assert_allclose(c, [30.0, 50.0])


def test_get_centroid_multiple():
    bboxes = np.array([[0, 0, 10, 10], [10, 10, 20, 20]])
    c = get_centroid(bboxes)
    assert c.shape == (2, 2)
    np.testing.assert_allclose(c, [[5.0, 5.0], [20.0, 20.0]])


def test_iou_identical_boxes():
    box = [0, 0, 10, 10]  # xyxy
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_partial_overlap():
    # boxes overlap on a 5x5 region; union = 100 + 100 - 25 = 175
    assert iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25.0 / 175.0)


def test_iou_xywh_matches_iou():
    # same boxes expressed in xywh
    assert iou_xywh([0, 0, 10, 10], [5, 5, 10, 10]) == pytest.approx(25.0 / 175.0)


def test_xyxy2xywh_and_back_single():
    xyxy = np.array([10, 20, 30, 50])
    xywh = xyxy2xywh(xyxy)
    # width/height include the +1 convention used in the implementation
    np.testing.assert_array_equal(xywh, [10, 20, 21, 31])


def test_xywh2xyxy_single():
    xywh = np.array([10, 20, 20, 30])
    xyxy = xywh2xyxy(xywh)
    np.testing.assert_array_equal(xyxy, [10, 20, 30, 50])


def test_xyxy2xywh_2d():
    xyxy = np.array([[0, 0, 9, 9], [10, 10, 29, 29]])
    xywh = xyxy2xywh(xyxy)
    np.testing.assert_array_equal(xywh, [[0, 0, 10, 10], [10, 10, 20, 20]])


def test_midwh2xywh_single():
    midwh = np.array([15, 25, 10, 20])
    xywh = midwh2xywh(midwh)
    np.testing.assert_array_equal(xywh, [10, 15, 10, 20])


def test_intersection_complement_indices():
    big = np.array([0, 1, 2, 3, 4])
    small = np.array([1, 3])
    result = intersection_complement_indices(big, small)
    np.testing.assert_array_equal(result, [0, 2, 4])


def test_nms_suppresses_overlapping_box():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]])
    scores = np.array([0.9, 0.8, 0.7])
    kept_boxes, kept_scores = nms(boxes, scores, overlapThresh=0.3)
    # the two highly-overlapping boxes collapse to one; the far box remains
    assert len(kept_boxes) == 2
    assert len(kept_scores) == 2
