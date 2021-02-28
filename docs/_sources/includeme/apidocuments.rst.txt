.. reference_docs:

Tracker
=======

.. autoclass:: motrackers.tracker.Tracker
    :members:

SORT
====

.. autofunction:: motrackers.sort_tracker.assign_tracks2detection_iou

.. autoclass:: motrackers.sort_tracker.SORT
    :members:

IOU Tracker
===========

.. autoclass:: motrackers.iou_tracker.IOUTracker
    :members:

Kalman Filter based Centroid Tracker
====================================

.. autofunction:: motrackers.centroid_kf_tracker.assign_tracks2detection_centroid_distances

.. autoclass:: motrackers.centroid_kf_tracker.CentroidKF_Tracker
    :members:

Tracks
======

.. autoclass:: motrackers.track.Track
    :members:

.. autoclass:: motrackers.track.KFTrackSORT
    :members:

.. autoclass:: motrackers.track.KFTrack4DSORT
    :members:

.. autoclass:: motrackers.track.KFTrackCentroid
    :members:

Kalman Filters
==============

.. autoclass:: motrackers.kalman_tracker.KalmanFilter
    :members:

.. autoclass:: motrackers.kalman_tracker.KFTrackerConstantAcceleration
    :members:

.. autoclass:: motrackers.kalman_tracker.KFTracker1D
    :members:

.. autoclass:: motrackers.kalman_tracker.KFTracker2D
    :members:

.. autoclass:: motrackers.kalman_tracker.KFTracker4D
    :members:

.. autoclass:: motrackers.kalman_tracker.KFTrackerSORT
    :members:

Object Detection
================

.. autoclass:: motrackers.detectors.detector.Detector
    :members:

.. autoclass:: motrackers.detectors.caffe.Caffe_SSDMobileNet
    :members:

.. autoclass:: motrackers.detectors.tf.TF_SSDMobileNetV2
    :members:

.. autoclass:: motrackers.detectors.yolo.YOLOv3
    :members:

Utilities
=========

.. autofunction:: motrackers.utils.misc.get_centroid

.. autofunction:: motrackers.utils.misc.iou

.. autofunction:: motrackers.utils.misc.iou_xywh

.. autofunction:: motrackers.utils.misc.xyxy2xywh

.. autofunction:: motrackers.utils.misc.xywh2xyxy

.. autofunction:: motrackers.utils.misc.midwh2xywh

.. autofunction:: motrackers.utils.misc.intersection_complement_indices

.. autofunction:: motrackers.utils.misc.nms

.. autofunction:: motrackers.utils.misc.draw_tracks

.. autofunction:: motrackers.utils.misc.load_labelsjson

.. autofunction:: motrackers.utils.misc.dict2jsonfile

.. autofunction:: motrackers.utils.filechooser_utils.create_filechooser

.. autofunction:: motrackers.utils.filechooser_utils.select_caffemodel_prototxt

.. autofunction:: motrackers.utils.filechooser_utils.select_caffemodel_weights

.. autofunction:: motrackers.utils.filechooser_utils.select_caffemodel

.. autofunction:: motrackers.utils.filechooser_utils.select_videofile

.. autofunction:: motrackers.utils.filechooser_utils.select_yolo_weights

.. autofunction:: motrackers.utils.filechooser_utils.select_coco_labels

.. autofunction:: motrackers.utils.filechooser_utils.select_yolo_config

.. autofunction:: motrackers.utils.filechooser_utils.select_yolo_model

.. autofunction:: motrackers.utils.filechooser_utils.select_pbtxt

.. autofunction:: motrackers.utils.filechooser_utils.select_tfmobilenet_weights

.. autofunction:: motrackers.utils.filechooser_utils.select_tfmobilenet

.. mdinclude:: ./../../../DOWNLOAD_WEIGHTS.md

.. mdinclude:: ./../../readme/REFERENCES.md

.. mdinclude:: ./../../readme/CODE_OF_CONDUCT.md
