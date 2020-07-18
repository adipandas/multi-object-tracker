
class Track:
    count = 0

    def __init__(self, track_id, centroid, bbox=None, class_id=None):
        """
        Track

        Parameters
        ----------
        track_id :
        centroid :
        bbox :
        class_id :
        """
        self.id = track_id
        self.class_id = class_id

        Track.count += 1

        self.centroid = centroid
        self.bbox = bbox
        self.lost = 0

        self.info = {}
