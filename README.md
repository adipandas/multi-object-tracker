[output_video]: ./assets/sample-output.gif "Sample Output"

# multi-object-tracker
object detection using deep learning and multi-object tracking

![Output Sample][output_video]

### Install OpenCV
Pip install for OpenCV (version 3.4.3 or later) is available [here](https://pypi.org/project/opencv-python/) and can be done with the following command:

`pip install opencv-contrib-python`

### Run with YOLO

1. Open the terminal
2. Go to `yolo_dir` in this repository: `cd ./yolo_dir`
3. Run: `sudo chmod +x ./get_yolo.sh`
4. Run: `./get_yolo.sh`

The model and the config files will be downloaded in `./yolo_dir`. These will be used `tracking-yolo-model.ipynb`.

- The video input can be specified in the cell named `Initiate opencv video capture object` in the notebook.
- To make the source as the webcam, use `video_src=0` else provide the path of the video file (example: `video_src="/path/of/videofile.mp4"`).

Example video used in above demo: https://flic.kr/p/L6qyxj

### Run with Caffemodel
- You have to use `tracking-caffe-model.ipynb`.
- The model for use is provided in the folder named `caffemodel_dir`.
- The video input can be specified in the cell named `Initiate opencv video capture object` in the notebook.
- To make the source as the webcam, use `video_src=0` else provide the path of the video file (example: `video_src="/path/of/videofile.mp4"`).

### References
The work here is based on the following literature available:
1. http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf
2. Pyimagesearch [1](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/), [2](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
3. [correlationTracker](https://github.com/Wenuka/correlationTracker)
4. [Caffemodel zoo](http://caffe.berkeleyvision.org/model_zoo.html)
5. [Caffemodel zoo GitHub](https://github.com/BVLC/caffe/tree/master/models)
6. [YOLO v3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

Use the caffemodel zoo from the reference [4,5] mentioned above to vary the CNN models and Play around with the codes.
