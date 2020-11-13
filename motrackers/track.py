import numpy as np


class Track:
    """
    Track

    Parameters
    ----------
    frame_id : int
        Camera frame id.
    track_id : int
        Track Id
    bbox : numpy.ndarray
        Bounding box pixel coordinates as (xmin, ymin, width, height) of the track.
    detection_confidence : float
        Detection confidence of the object (probability).
    class_id : Object
        Class label id.
    lost : int
        Number of times the object or track was not tracked by tracker in consecutive frames.
    iou_score : float
        Intersection over union score.
    kwargs : dict
        Additional key word arguments.
    """

    count = 0

    metadata = dict(
        data_output_formats=['mot_challenge', 'visdrone_challenge']
    )

    def __init__(
        self,
        track_id,
        frame_id,
        bbox,
        detection_confidence,
        class_id=None,
        lost=0,
        iou_score=0.,
        data_output_format='mot_challenge',
        **kwargs
    ):
        assert data_output_format in Track.metadata['data_output_formats']
        Track.count += 1
        self.id = track_id

        self.detection_confidence_max = 0.
        self.lost = 0
        self.age = 0

        self.update(frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs)

        if data_output_format == 'mot_challenge':
            self.output = self.get_mot_challenge_format
        elif data_output_format == 'visdrone_challenge':
            self.output = self.get_vis_drone_format
        else:
            raise NotImplementedError

    def update(self, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0., **kwargs):
        """
        Update the track.

        Parameters
        ----------
        frame_id : int
            Camera frame id.
        bbox : numpy.ndarray
            Bounding box pixel coordinates as (xmin, ymin, width, height) of the track.
        detection_confidence : float
            Detection confidence of the object (probability).
        class_id : Object
            Class label id.
        lost : int
            Number of times the object or track was not tracked by tracker in consecutive frames.
        iou_score : float
            Intersection over union score.
        kwargs : dict
            Additional key word arguments.

        Returns
        -------

        """
        self.class_id = class_id
        self.bbox = np.array(bbox)
        self.detection_confidence = detection_confidence
        self.frame_id = frame_id
        self.lost = lost
        self.iou_score = iou_score

        if lost == 0:
            self.lost = 0
        else:
            self.lost += lost

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.detection_confidence_max = max(self.detection_confidence_max, detection_confidence)

        self.age += 1

    @property
    def centroid(self):
        """
        Return the centroid of the bounding box.

        Returns
        -------
        numpy.ndarray: Centroid (x, y) of bounding box.
        """
        return (self.bbox[0] + 0.5 * self.bbox[2]), (self.bbox[1] + 0.5 * self.bbox[3])

    def get_mot_challenge_format(self):
        """
        Get the tracker data in MOT challenge format as a tuple of elements containing
        `(frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)`

        References
        ----------
        - Website : https://motchallenge.net/

        Returns
        -------
        mot_tuple : tuple
            Tuple of 10 elements representing `(frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)`.
        """
        mot_tuple = (
            self.frame_id, self.id, self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], self.detection_confidence,
            -1, -1, -1
        )
        return mot_tuple

    def get_vis_drone_format(self):
        """
        Track data output in VISDRONE Challenge format with tuple as
        `(frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category,
        truncation, occlusion)`.

        References
        ----------
        - Website : http://aiskyeye.com/
        - Paper : https://arxiv.org/abs/2001.06303
        - GitHub : https://github.com/VisDrone/VisDrone2018-MOT-toolkit
        - GitHub : https://github.com/VisDrone/

        Returns
        -------
        mot_tuple : tuple
            Tuple containing the elements as `(frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
            score, object_category, truncation, occlusion)`.
        """
        mot_tuple = (
            self.frame_id, self.id, self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3],
            self.detection_confidence, self.class_id, -1, -1
        )
        return mot_tuple

    @staticmethod
    def print_all_track_output_formats():
        print(Track.metadata['data_output_formats'])
