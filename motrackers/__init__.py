"""
Multi-object Trackers in Python:
    - GitHub link: https://github.com/adipandas/multi-object-tracker
    - Author: Aditya M. Deshpande
    - Blog: http://adipandas.github.io/
"""


from motrackers.tracker import Tracker as CentroidTracker
from motrackers.centroid_kf_tracker import CentroidKF_Tracker
from motrackers.sort_tracker import SORT
from motrackers.iou_tracker import IOUTracker
