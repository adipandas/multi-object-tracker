

def get_centroid(bounding_box):
    """
    Calculate the centroid of bounding box.

    :param bounding_box: list of bounding box coordinates of top-left and bottom-right (xlt, ylt, xrb, yrb)
    :return: bounding box centroid coordinates (x, y)
    """

    xlt, ylt, xrb, yrb = bounding_box
    centroid_x = int((xlt + xrb) / 2.0)
    centroid_y = int((ylt + yrb) / 2.0)

    return centroid_x, centroid_y
