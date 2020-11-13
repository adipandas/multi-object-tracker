import numpy as np
import cv2 as cv


def get_centroid(bboxes):
    """
    Calculate centroids for multiple bounding boxes.

    Parameters
    ----------
    bboxes : numpy.ndarray
        Array of shape `(n, 4)` or of shape `(4,)`.
        Where each row contains `(xmin, ymin, xmax, ymax)`.

    Returns
    -------
    numpy.ndarray : Centroid (x, y) coordinates of shape `(n, 2)` or `(2,)`.

    """
    if len(bboxes.shape) == 2:
        assert bboxes.shape[1] == 4, "Input shape is {}, expecting shape[1]==4".format(bboxes.shape)
        x = np.mean(bboxes[:, [0, 2]], axis=1, keepdims=True, dtype='int')
        y = np.mean(bboxes[:, [1, 3]], axis=1, keepdims=True, dtype='int')
        centroids = np.concatenate([x, y], axis=1)
        return centroids
    elif len(bboxes.shape) == 1:
        assert bboxes.shape[0] == 4, "Input shape is {}, expecting shape[0]==4".format(bboxes.shape)
        x = 0.5*(bboxes[0] + bboxes[2])
        y = 0.5*(bboxes[1] + bboxes[3])
        return np.array([x, y])


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Source: https://github.com/bochinski/iou-tracker/blob/master/util.py

    Parameters
    ----------
    bbox1 : numpy.array, list of floats
            bounding box in format (x-top-left, y-top-left, x-bottom-right, y-bottom-right) of length 4.
    bbox2 : numpy.array, list of floats
            bounding box in format (x-top-left, y-top-left, x-bottom-right, y-bottom-right) of length 4.

    Returns
    -------
    iou: float
         intersection-over-onion of bbox1, bbox2.
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1), (x0_2, y0_2, x1_2, y1_2) = bbox1, bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0.0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    iou_ = size_intersection / size_union

    return iou_


def xyxy2xywh(xyxy):
    """
    Convert bounding box coordinates from (xmin, ymin, xmax, ymax) format to (xmin, ymin, width, height).

    Parameters
    ----------
    xyxy : numpy.ndarray

    Returns
    -------
    numpy.ndarray : Bounding box coordinates (xmin, ymin, width, height).

    """

    if len(xyxy.shape) == 2:
        w, h = xyxy[:, 2] - xyxy[:, 0] + 1, xyxy[:, 3] - xyxy[:, 1] + 1
        xywh = np.concatenate((xyxy[:, 0:2], w[:, None], h[:, None]), axis=1)
        return xywh.astype("int")
    elif len(xyxy.shape) == 1:
        (left, top, right, bottom) = xyxy
        width = right - left + 1
        height = bottom - top + 1
        return np.array([left, top, width, height]).astype('int')
    else:
        raise ValueError("Input shape not compatible.")


def xywh2xyxy(xywh):
    """
    Convert bounding box coordinates from (xmin, ymin, width, height) to (xmin, ymin, xmax, ymax) format.

    Parameters
    ----------
    xywh : numpy.ndarray
        Bounding box coordinates as (xmin, ymin, width, height)

    Returns
    -------
    numpy.ndarray : bounding box coordinates as (xmin, ymin, xmax, ymax)

    """

    if len(xywh.shape) == 2:
        x = xywh[:, 0] + xywh[:, 2]
        y = xywh[:, 1] + xywh[:, 3]
        xyxy = np.concatenate((xywh[:, 0:2], x[:, None], y[:, None]), axis=1).astype('int')
        return xyxy
    if len(xywh.shape) == 1:
        x, y, w, h = xywh
        xr = x + w
        yb = y + h
        return np.array([x, y, xr, yb]).astype('int')


def midwh2xywh(midwh):
    """
    Convert bounding box coordinates from (xmid, ymid, width, height) to (xmin, ymin, width, height) format.

    Parameters
    ----------
    midwh : numpy.ndarray
        Bounding box coordinates (xmid, ymid, width, height)

    Returns
    -------
    numpy.ndarray : Bounding box coordinates (xmin, ymin, width, height).
    """

    if len(midwh.shape) == 2:
        xymin = midwh[:, 0:2] - midwh[:, 2:] * 0.5
        wh = midwh[:, 2:]
        xywh = np.concatenate([xymin, wh], axis=1).astype('int')
        return xywh
    if len(midwh.shape) == 1:
        xmid, ymid, w, h = midwh
        xywh = np.array([xmid-w*0.5, ymid-h*0.5, w, h]).astype('int')
        return xywh


def intersection_complement_indices(big_set_indices, small_set_indices):
    """
    Get the complement of intersection of two sets of indices.

    Parameters
    ----------
    big_set_indices :  numpy.ndarray
        Indices of big set.
    small_set_indices : numpy.ndarray
        Indices of small set.

    Returns
    -------
    intersection_complement : numpy.ndarray
        Indices of set which is complementary to intersection of two input sets.

    """
    assert big_set_indices.shape[0] >= small_set_indices.shape[1]
    n = len(big_set_indices)
    mask = np.ones((n,), dtype=bool)
    mask[small_set_indices] = False
    intersection_complement = big_set_indices[mask]
    return intersection_complement


def nms(boxes, scores, overlapThresh, classes=None):
    """
    Non-maximum suppression. based on Malisiewicz et al.

    Args:
        boxes (numpy.ndarray): Boxes to process (xmin, ymin, xmax, ymax)
        scores (numpy.ndarray): Corresponding scores for each box
        overlapThresh (float):  Overlap threshold for boxes to merge
        classes (numpy.ndarray, optional): Class ids for each box.

    Returns:
        (tuple): a tuple containing:
            - boxes (list): nms boxes
            - scores (list): nms scores
            - classes (list, optional): nms classes if specified

    """

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    if scores.dtype.kind == "i":
        scores = scores.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    if classes is not None:
        return boxes[pick], scores[pick], classes[pick]
    else:
        return boxes[pick], scores[pick]


def draw_tracks(image, tracks):
    """
    Draw on input image.

    Args:
        image (numpy.ndarray): image
        tracks (list): list of tracks to be drawn on the image.

    Returns:
        numpy.ndarray : image with the track-ids drawn on it.
    """

    for trk in tracks:

        trk_id = trk[1]
        xmin = trk[2]
        ymin = trk[3]
        width = trk[4]
        height = trk[5]

        xcentroid, ycentroid = int(xmin + 0.5*width), int(ymin + 0.5*height)

        text = "ID {}".format(trk_id)

        cv.putText(image, text, (xcentroid - 10, ycentroid - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(image, (xcentroid, ycentroid), 4, (0, 255, 0), -1)

    return image


if __name__ == '__main__':
    bb = np.random.random_integers(0, 100, size=(20,)).reshape((5, 4))
    c = get_centroid(bb)
    print(bb, c)
    
    bb2 = np.array([1, 2, 3, 4])
    c2 = get_centroid(bb2)
    print(bb2, c2)
