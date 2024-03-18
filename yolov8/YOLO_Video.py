from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from flask import jsonify
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.nn.tasks import Ensemble
import torch
import yaml
import numpy as np

modelDefault = YOLO("./weights/yolov8n-pose.pt")
classNames = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

classNamesOpenImageV7 = []
with open("./weights/OpenImagesV7.yaml", "r", encoding="utf-8") as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)
    classNamesOpenImageV7 = [data["names"][i] for i in sorted(data["names"])]

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

modelPersonPose = YOLO("./weights/yolov8s-pose.pt").to(device)
modelObjectDetection = YOLO("./weights/yolov8x.pt").to(device)

tracker = DeepSort(max_age=40, embedder_gpu=True, max_iou_distance=0.7)

tracks = []


def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture, cv2.CAP_GSTREAMER)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    while cap.isOpened():
        success, img = cap.read()

        print("=========", img)

        if not success:
            print("Error: Unable to read frame.")
            break

        results = modelPersonPose(img, stream=True)
        results2 = modelObjectDetection(
            img,
            stream=True,
        )

        detect = []
        for r in results2:
            box = r.boxes
            for i, xyxy in enumerate(box.xyxy):
                x1, y1, x2, y2 = xyxy
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = box.conf[i]
                cls = int(box.cls[i])
                class_name = classNames[cls]

                if (
                    class_name != "person"
                    and class_name != "bottle"
                    # and class_name != "Wine"
                    # and class_name != "Man"
                    # and class_name != "Woman"
                ):
                    continue

                label = f"{class_name}:{conf}:{conf}"
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]

                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                # cv2.rectangle(
                #     img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA
                # )  # filled
                # cv2.putText(
                #     img,
                #     label,
                #     (x1, y1 - 2),
                #     0,
                #     1,
                #     [255, 255, 255],
                #     thickness=1,
                #     lineType=cv2.LINE_AA,
                # )

                detect.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])

        tracks = tracker.update_tracks(detect, frame=img)
        for track in tracks:
            print("track: ", track)
            if track.is_confirmed():
                track_id = track.track_id

                leftTopRightButton = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, leftTopRightButton)

                class_name = classNames[int(class_id)]

                label = f"{class_name}:{track_id}"

                cv2.rectangle(img, (x1, y1), (x2, y2), (148, 0, 211), 2)
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 5),
                    0,
                    1,
                    [148, 0, 211],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        #
        #
        #  pose
        for r in results:
            index = 0
            names = r.names
            boxes = r.boxes
            keypointsRawData = r.keypoints.data

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = names[cls]
                label = f"{class_name}"
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                # cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                # cv2.putText(
                #     img,
                #     label,
                #     (x1, y1 - 2),
                #     0,
                #     1,
                #     [255, 255, 255],
                #     thickness=1,
                #     lineType=cv2.LINE_AA,
                # )

            for keypointRawData in keypointsRawData:
                for i, keypoint in enumerate(keypointRawData):
                    x, y = int(keypoint[0].item()), int(keypoint[1].item())

                    cv2.putText(
                        img,
                        str(i),
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        1,
                    )

        yield img


cv2.destroyAllWindows()


def handleDetect(img):
    results = modelDefault(img, stream=True)
    detections = []
    for r in results:
        index = 0
        detections = []

        names = r.names
        boxes = r.boxes
        keypointsRawData = r.keypoints.data

        for box in boxes:
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = names[cls]
            xywh = (xyxy2xywh(torch.tensor(box.xyxy[0]).view(1, 4))).view(-1).tolist()
            x, y, w, h = xywh
            x = x - (w / 2)  # Calculate top left x
            y = y - (h / 2)  # Calculate top left y

            detections.insert(
                index,
                {
                    "tag": label,
                    "confidence": float(conf),
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                },
            )

        points = []
        for keypointRawData in keypointsRawData:
            for i, keypoint in enumerate(keypointRawData):
                x, y = int(keypoint[0].item()), int(keypoint[1].item())

                points.insert(
                    index,
                    {
                        "index": i,
                        "x": x,
                        "y": y,
                    },
                )

        detections[index]["points"] = points
        index += 1

    return detections
