[cars-yolo-output]: examples/assets/cars.gif "Sample Output with YOLO"
[cows-tf-ssd-output]: examples/assets/cows.gif "Sample Output with SSD"

# Multi-object trackers in Python
Easy to use implementation of various multi-object tracking algorithms.

[![DOI](https://zenodo.org/badge/148338463.svg)](https://zenodo.org/badge/latestdoi/148338463)


`YOLOv3 + CentroidTracker` |  `TF-MobileNetSSD + CentroidTracker`
:-------------------------:|:-------------------------:
![Cars with YOLO][cars-yolo-output]  |  ![Cows with tf-SSD][cows-tf-ssd-output]
Video source: [link](https://flic.kr/p/L6qyxj) | Video source: [link](https://flic.kr/p/26WeEWy)

## Available Multi Object Trackers

```
CentroidTracker
IOUTracker
CentroidKF_Tracker
SORT
```

## Available OpenCV-based object detectors:

```
detector.TF_SSDMobileNetV2
detector.Caffe_SSDMobileNet
detector.YOLOv3
```

## Installation

Pip install for OpenCV (version 3.4.3 or later) is available [here](https://pypi.org/project/opencv-python/) and can be done with the following command:

```
git clone https://github.com/adipandas/multi-object-tracker
cd multi-object-tracker
pip install -r requirements.txt
pip install -e .
```

**Note - for using neural network models with GPU**  
For using the opencv `dnn`-based object detection modules provided in this repository with GPU, you may have to compile a CUDA enabled version of OpenCV from source.  
* To build opencv from source, refer the following links:
[[link-1](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html)],
[[link-2](https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/)]

## How to use?: Examples

The interface for each tracker is simple and similar. Please refer the example template below.

```
from motrackers import CentroidTracker # or IOUTracker, CentroidKF_Tracker, SORT
input_data = ...
detector = ...
tracker = CentroidTracker(...) # or IOUTracker(...), CentroidKF_Tracker(...), SORT(...)
while True:
    done, image = <read(input_data)>
    if done:
        break
    detection_bboxes, detection_confidences, detection_class_ids = detector.detect(image)
    # NOTE: 
    # * `detection_bboxes` are numpy.ndarray of shape (n, 4) with each row containing (bb_left, bb_top, bb_width, bb_height)
    # * `detection_confidences` are numpy.ndarray of shape (n,);
    # * `detection_class_ids` are numpy.ndarray of shape (n,).
    output_tracks = tracker.update(detection_bboxes, detection_confidences, detection_class_ids)
    # `output_tracks` is a list with each element containing tuple of
    # (<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>)
    for track in output_tracks:
        frame, id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z = track
        assert len(track) == 10
        print(track)
```

Please refer [examples](https://github.com/adipandas/multi-object-tracker/tree/master/examples) folder of this repository for more details. You can clone and run the examples.

## Pretrained object detection models

You will have to download the pretrained weights for the neural-network models. 
The shell scripts for downloading these are provided [here](https://github.com/adipandas/multi-object-tracker/tree/master/examples/pretrained_models) below respective folders.
Please refer [DOWNLOAD_WEIGHTS.md](https://github.com/adipandas/multi-object-tracker/blob/master/DOWNLOAD_WEIGHTS.md) for more details.

### Notes
* There are some variations in implementations as compared to what appeared in papers of `SORT` and `IoU Tracker`.
* In case you find any bugs in the algorithm, I will be happy to accept your pull request or you can create an issue to point it out.

## References, Credits and Contributions
Please see [REFERENCES.md](https://github.com/adipandas/multi-object-tracker/blob/master/docs/readme/REFERENCES.md) and [CONTRIBUTING.md](https://github.com/adipandas/multi-object-tracker/blob/master/docs/readme/CONTRIBUTING.md).

## Citation

If you use this repository in your work, please consider citing it with:
```
@misc{multiobjtracker_amd2018,
  author = {Deshpande, Aditya M.},
  title = {Multi-object trackers in Python},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/adipandas/multi-object-tracker}},
}
```

