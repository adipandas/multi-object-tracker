
class Track:
    count = 0

    def __init__(self, track_id, centroid, bbox=None, class_id=None):
        """
        Track

        Parameters
        ----------
        track_id : int
                   Track id.
        centroid : tuple
                   Centroid of the track pixel coordinate (x, y).
        bbox : tuple, list, numpy.ndarray
               Bounding box of the track.
        class_id : int
                   Class label id.
        """
        self.id = track_id
        self.class_id = class_id

        Track.count += 1

        self.centroid = centroid
        self.bbox = bbox
        self.lost = 0

        self.info = dict(
            max_score=0.0,
            lost=0,
            score=0.0,
        )
