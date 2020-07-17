from collections import OrderedDict
import numpy as np
from scipy.spatial import distance


class SimpleTracker:
    """
    Greedy Tracker
    """
    def __init__(self, max_lost=30):
        """
        :param max_lost: maximum number of consecutive frames object was not detected.
        :type max_lost: int
        """
        self.nextObjectID = 0               # ID of next object
        self.objects = OrderedDict()        # stores ID:Locations
        self.lost = OrderedDict()           # stores ID:Lost_count
        self.maxLost = max_lost

    def addObject(self, new_object_location):
        """
        Add a newly detected object to the queue

        :param new_object_location: centroid coordinate (x, y) in pixels of the bounding box
        """

        # store new object location
        self.objects[self.nextObjectID] = new_object_location

        # initialize frame_counts for when new object is undetected
        self.lost[self.nextObjectID] = 0

        self.nextObjectID += 1

    def removeObject(self, objectID):
        """
        Remove tracker data after object is lost

        :param objectID: ID of the object lost from tracker
        """
        del self.objects[objectID]
        del self.lost[objectID]

    @staticmethod
    def getLocation(bounding_box):
        """
        Calculate the centroid of bounding box.

        :param bounding_box: list of bounding box coordinates top-left and bottom-right (xlt, ylt, xrb, yrb)
        :return: bounding box centroid coordinates (x, y)
        """

        xlt, ylt, xrb, yrb = bounding_box
        return int((xlt + xrb) / 2.0), int((ylt + yrb) / 2.0)

    def update(self, detections):
        """
        Update the tracker based on the new detections as input.

        :param detections: list of bounding box coordinates of detected objects in the image.
            Each element of the list is a tuple of pixel coordinates of the following form
            (top-left-x, top-left-y, bottom-right-x, bottom-right-y).
        :return: dictionary of objects with key as object-id and
            value as centroid-pixel-coordinates (x, y) of that object.
        """

        # if no object detected in the frame
        if len(detections) == 0:
            lost_ids = list(self.lost.keys())
            for objectID in lost_ids:
                self.lost[objectID] += 1
                if self.lost[objectID] > self.maxLost:
                    self.removeObject(objectID)

            return self.objects

        # current object locations
        new_object_locations = np.zeros((len(detections), 2), dtype="int")
        for (i, detection) in enumerate(detections):
            new_object_locations[i] = self.getLocation(detection)

        if len(self.objects):
            objectIDs = list(self.objects.keys())
            previous_object_locations = np.array(list(self.objects.values()))

            # pairwise distance between previous and current
            D = distance.cdist(previous_object_locations, new_object_locations)

            # (minimum distance of previous from current).sort_as_per_index
            row_idx = D.min(axis=1).argsort()           # old object idx

            # index of minimum distance of previous from current
            col_idx = D.argmin(axis=1)[row_idx]         # new object idx sorted as per distance from old object ids

            assignedRows, assignedCols = set(), set()
            for (row, col) in zip(row_idx, col_idx):

                if row in assignedRows or col in assignedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = new_object_locations[col]
                self.lost[objectID] = 0

                assignedRows.add(row)
                assignedCols.add(col)

            unassignedRows = set(range(0, D.shape[0])).difference(assignedRows)
            unassignedCols = set(range(0, D.shape[1])).difference(assignedCols)

            if D.shape[0] >= D.shape[1]:
                # length of old-detections is more than new-detections
                for row in unassignedRows:
                    objectID = objectIDs[row]
                    self.lost[objectID] += 1

                    if self.lost[objectID] > self.maxLost:
                        self.removeObject(objectID)
            else:
                for col in unassignedCols:
                    self.addObject(new_object_locations[col])

        else:
            for i in range(0, len(detections)):
                self.addObject(new_object_locations[i])

        return self.objects
