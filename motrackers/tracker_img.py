import argparse
import time
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default='../examples/video_data/cars.mp4', help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
args = vars(ap.parse_args())

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

trackers = cv2.MultiTracker_create()

vs = cv2.VideoCapture(args["video"])

while True:
    ok, frame = vs.read()
    if not ok:
        break

    # resize the frame (so we can process it faster)
    frame = cv2.resize(frame, (600, 400))

    (success, boxes) = trackers.update(frame)
    print(success)

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to tracks
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

        # create a new object tracker for the bounding box and add it to our multi-object tracker
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        trackers.add(tracker, frame, box)

    elif key == ord("q"):  # if the `q` key was pressed, break from the loop
        break

    time.sleep(0.1)

# if we are using a webcam, release the pointer
vs.release()

# close all windows
cv2.destroyAllWindows()
