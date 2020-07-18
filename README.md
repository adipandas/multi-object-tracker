[cars-yolo-output]: ./assets/cars.gif "Sample Output with YOLO"
[cows-tf-ssd-output]: ./assets/cows.gif "Sample Output with SSD"

# multi-object-tracker
Object detection using deep learning and multi-object tracking

[![DOI](https://zenodo.org/badge/148338463.svg)](https://zenodo.org/badge/latestdoi/148338463)


#### YOLO
Video Source: [link](https://flic.kr/p/89KYXt)

![Cars with YOLO][cars-yolo-output]

#### Tensorflow-SSD-MobileNet
Video Source: [link](https://flic.kr/p/26WeEWy)

![Cows with tf-SSD][cows-tf-ssd-output]


### Installation
Pip install for OpenCV (version 3.4.3 or later) is available [here](https://pypi.org/project/opencv-python/) and can be done with the following command:

```
pip install numpy matplotlib scipy
pip install opencv-contrib-python
```

Installation of `ipyfilechooser` is recommended if you want to use the jupyter notebooks available in the ```examples``` folder.
```
pip install ipyfilechooser
```

```
git clone https://github.com/adipandas/multi-object-tracker
cd multi-object-tracker
pip install -e .
```

### YOLO

Do the following in the terminal:
```
cd ./pretrained_models/yolo_weights
sudo chmod +x ./get_yolo.sh
./get_yolo.sh
```

The above commands will download the model and the config files in `./pretrained_models/yolo_weights`.
These weights are to be used in `examples/tracking-yolo-model.ipynb`.

- The video input can be specified in the cell named `Initiate opencv video capture object` in the notebook.
- To make the source as the webcam, use `video_src=0` else provide the path of the video file (example: `video_src="/path/of/videofile.mp4"`).

Example video used in above demo was taken from [here](https://flic.kr/p/L6qyxj)

### TensorFlow model

Do the following in the terminal:
```
cd ./pretrained_models/tensorflow_weights
sudo chmod +x ./get_ssd_model.sh
./get_ssd_model.sh
```

This will download model and config files in `./pretrained_models/tensorflow_weights`.
These will be used `examples/tracking-tensorflow-ssd_mobilenet_v2_coco_2018_03_29.ipynb`.

**SSD-Mobilenet_v2_coco_2018_03_29** was used for this example.
Other networks can be downloaded and ran: Go through `tracking-tensorflow-ssd_mobilenet_v2_coco_2018_03_29.ipynb` for more details.

- The video input can be specified in the cell named `Initiate opencv video capture object` in the notebook.
- To make the source as the webcam, use `video_src=0` else provide the path of the video file (example: `video_src="/path/of/videofile.mp4"`).

Video used in SSD-Mobilenet multi-object detection and tracking can be found [here](https://flic.kr/p/89KYXt)

### Caffemodel

Do the following in the terminal
```
cd ./pretrained_models/caffemodel_weights
sudo chmod +x ./get_caffemodel.sh
./get_caffemodel.sh
```

This will download model and config files in `./pretrained_models/caffemodel_weights`.
These will be used `examples/tracking-caffe-model-mobilenetSSD.ipynb`.

The caffemodel example provided here also uses MobileNet-SSD model for detection.

### References and Credits
This work is based on the following literature:
1. Bochinski, E., Eiselein, V., & Sikora, T. (2017, August). High-speed tracking-by-detection without using image information. In 2017 14th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS) (pp. 1-6). IEEE. [[paper-pdf](http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf)]
2. Pyimagesearch [link-1](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/), [link-2](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
3. [correlationTracker](https://github.com/Wenuka/correlationTracker)
4. [Caffemodel zoo](http://caffe.berkeleyvision.org/model_zoo.html)
5. [Caffemodel zoo GitHub](https://github.com/BVLC/caffe/tree/master/models)
6. [YOLO v3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

Use the caffemodel zoo from the reference [4,5] mentioned above to vary the CNN models and Play around with the codes.

***Suggestion**: If you are looking for speed go for SSD-mobilenet. If you are looking for accurracy and speed go with YOLO. The best way is to train and fine tune your models on your dataset. Although, Faster-RCNN gives more accurate object detections, you will have to compromise on the detection speed as it is slower as compared to YOLO.*

### Citation

If you use this repository in your work, please consider citing it with:
```
@misc{multiobjtracker_amd2018,
  author = {Deshpande, Aditya M.},
  title = {Multi-object trackers in Python},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/adipandas/multi-object-tracker}},
}
```

```
@software{aditya_m_deshpande_2019_3530936,
  author       = {Aditya M. Deshpande},
  title        = {{adipandas/multi-object-tracker: multi-object- 
                   tracker}},
  month        = nov,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {v4.0.0},
  doi          = {10.5281/zenodo.3530936},
  url          = {https://doi.org/10.5281/zenodo.3530936}
}
```
