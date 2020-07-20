import numpy as np


def get_centroid(bounding_box):
    """
    Calculate the centroid of bounding box.

    Parameters
    ----------
    bounding_box : list
                   list of bounding box coordinates of top-left and bottom-right (xlt, ylt, xrb, yrb)

    Returns
    -------
    centroid: tuple
              Bounding box centroid pixel coordinates (x, y).

    """
    xlt, ylt, xrb, yrb = bounding_box
    centroid_x = int((xlt + xrb) / 2.0)
    centroid_y = int((ylt + yrb) / 2.0)

    return centroid_x, centroid_y


def get_centroids(bboxes):
    if len(bboxes):
        assert bboxes.shape[1] == 4, "Input shape is {}, expecting shape[1]==4".format(bboxes.shape)

        x = np.mean(bboxes[:, [0, 2]], axis=1, keepdims=True, dtype='int')
        y = np.mean(bboxes[:, [1, 3]], axis=1, keepdims=True, dtype='int')
        centroids = np.concatenate([x, y], axis=1)
        return centroids
    else:
        return bboxes


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


if __name__ == '__main__':
    bb = np.random.random_integers(0, 100, size=(20,)).reshape((5, 4))
    c = get_centroids(bb)
    print(bb)
    print(c)
