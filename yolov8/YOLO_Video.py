from ultralytics import YOLO
import cv2
import math
from flask import jsonify
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.nn.tasks import Ensemble
import torch
import yaml

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


modelPersonPose = YOLO("./weights/yolov8n-pose.pt")
modelObjectDetection = YOLO("./weights/yolov8x-oiv7.pt")


def video_detection(path_x):
    video_capture = path_x
    # Create a Webcam Object
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))

    while True:
        success, img = cap.read()
        results = modelPersonPose(img, stream=True)
        results2 = modelObjectDetection(img, stream=True)

        for r in results2:
            # print(r.boxes)
            box = r.boxes
            for i, xyxy in enumerate(box.xyxy):
                x1, y1, x2, y2 = xyxy
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # print(xyxy)
                conf = box.conf[i]
                cls = int(box.cls[i])
                class_name = classNamesOpenImageV7[cls]

                # if class_name != "Bottle" and class_name != "Wine":
                #     continue

                label = f"{class_name}"
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]

                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.rectangle(
                    img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA
                )  # filled
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 2),
                    0,
                    1,
                    [255, 255, 255],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        for r in results:
            index = 0
            names = r.names
            boxes = r.boxes
            keypointsRawData = r.keypoints.data

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # print(x1,y1,x2,y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = names[cls]
                label = f"{class_name}"
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                # print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(
                    img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA
                )  # filled
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 2),
                    0,
                    1,
                    [255, 255, 255],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

                xywh = (
                    (xyxy2xywh(torch.tensor(box.xyxy[0]).view(1, 4))).view(-1).tolist()
                )
                myX, myY, w, h = xywh
                myX = myX - (w / 2)  # Calculate top left x
                myY = myY - (h / 2)  # Calculate top left y

            points = []
            for keypointRawData in keypointsRawData:
                for i, keypoint in enumerate(keypointRawData):
                    x, y = int(keypoint[0].item()), int(keypoint[1].item())
                    # cv2.circle(img, (x,y), 1,(255, 0, 0), 5)

                    points.insert(
                        index,
                        {
                            "index": i,
                            "x": x,
                            "y": y,
                        },
                    )

                    cv2.putText(
                        img,
                        str(i),
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
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
