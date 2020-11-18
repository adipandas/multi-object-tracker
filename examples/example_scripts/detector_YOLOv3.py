import cv2 as cv
from motrackers.detectors import YOLOv3


def main(video_path, model):
    cap = cv.VideoCapture(video_path)
    while True:
        ok, image = cap.read()

        if not ok:
            print("Cannot read the video feed.")
            break

        bboxes, confidences, class_ids = model.detect(image)
        updated_image = model.draw_bboxes(image, bboxes, confidences, class_ids)

        cv.imshow("image", updated_image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Object detections in input video using YOLOv3 trained on COCO dataset.'
    )

    parser.add_argument(
        '--video', '-v', type=str, default="./../video_data/cars.mp4", help='Input video path.')

    parser.add_argument(
        '--weights', '-w', type=str,
        default="./../pretrained_models/yolo_weights/yolov3.weights",
        help='path to weights file of YOLOv3 (`.weights` file.)'
    )

    parser.add_argument(
        '--config', '-c', type=str,
        default="./../pretrained_models/yolo_weights/yolov3.cfg",
        help='path to config file of YOLOv3 (`.cfg` file.)'
    )

    parser.add_argument(
        '--labels', '-l', type=str,
        default="./../pretrained_models/yolo_weights/coco_names.json",
        help='path to labels file of coco dataset (`.names` file.)'
    )

    parser.add_argument(
        '--gpu', type=bool,
        default=False, help='Flag to use gpu to run the deep learning model. Default is `False`'
    )

    args = parser.parse_args()

    model = YOLOv3(
        weights_path=args.weights,
        configfile_path=args.config,
        labels_path=args.labels,
        confidence_threshold=0.5,
        nms_threshold=0.2,
        draw_bboxes=True,
        use_gpu=args.gpu
    )

    main(args.video, model)
