from ultralytics import YOLO
import cv2
from flask import jsonify
from ultralytics.utils.ops import xyxy2xywh
import torch
import math
import yaml
import numpy as np
import threading
import tensorflow
import queue

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
with open("./yaml/OpenImagesV7.yaml", "r", encoding="utf-8") as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)
    classNamesOpenImageV7 = [data["names"][i] for i in sorted(data["names"])]

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

modelPersonPose = YOLO("./weights/yolov8s-pose.pt").to(device)
modelObjectDetection = YOLO("./weights/yolov8x.pt").to(device)

lstm_model = tensorflow.keras.models.load_model("./LSTM/results/lstm01.keras")

time_steps = 15


def video_detection(path_x):
    time_steps_ltsm = []

    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            print("Error: Unable to read frame.")
            continue

        modelPersonPoseResults = modelPersonPose.track(
            img,
            stream=False,
            persist=True,
            tracker="bytetrack.yaml",
        )
        modelObjectDetectionResults = modelObjectDetection.track(
            source=img,
            stream=False,
            persist=True,
            tracker="bytetrack.yaml",
        )

        for r in modelObjectDetectionResults:
            if r.boxes is None or r.boxes.id is None:
                continue

            box = r.boxes
            ids = r.boxes.id.cpu().numpy().astype(int)
            for i, xyxy in enumerate(box.xyxy):
                x1, y1, x2, y2 = xyxy
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[i] * 100)) / 100
                class_id = int(box.cls[i])
                class_name = classNames[class_id]
                label = f"{ids[i]}:{class_name}_{conf}"

                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                x, y, w, h = xywh
                x = x - (w / 2)
                y = y - (h / 2)

                if (
                    class_name
                    != "bottle"
                    # and class_name != "person"
                    #     # and class_name != "Wine"
                    #     # and class_name != "Man"
                    #     # and class_name != "Woman"
                ):
                    continue

                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
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

        #  pose
        for r in modelPersonPoseResults:
            if r.boxes is None or r.boxes.id is None:
                continue

            names = r.names
            boxes = r.boxes
            ids = r.boxes.id.cpu().numpy().astype(int)

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                class_id = int(box.cls[0])
                class_name = names[class_id]
                label = f"{ids[0]}:{class_name}_{conf}"
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
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

            keypoints_xy = r.keypoints.xy[0]
            for i, keypoint_tensor in enumerate(keypoints_xy):
                x, y = (
                    int(keypoint_tensor[0].item()),
                    int(keypoint_tensor[1].item()),
                )

                cv2.putText(
                    img,
                    str(i),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    1,
                )

            keypoints_xyn = r.keypoints.xyn[0]
            lstm_label = "NOTHING"
            items_ltsm = []
            for i, keypoint_tensor in enumerate(keypoints_xyn):
                x, y = (
                    keypoint_tensor[0].item(),
                    keypoint_tensor[1].item(),
                )

                items_ltsm.append(x)
                items_ltsm.append(y)

                time_steps_ltsm.append(items_ltsm)
                if len(time_steps_ltsm) == time_steps:
                    result_queue = queue.Queue()
                    lstmDetect_thread = threading.Thread(
                        target=lstmDetect,
                        args=(
                            lstm_model,
                            time_steps_ltsm,
                            result_queue,
                        ),
                    )
                    lstmDetect_thread.start()

                    lstmDetect_thread.join()
                    lstm_label = result_queue.get()

                    time_steps_ltsm = []
                    items_ltsm = []

                print("lstm_label lstm_label:", lstm_label)

        yield img

    cv2.destroyAllWindows()


def lstmDetect(model, lm_list, result_queue):
    print("results=============:", lm_list)

    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    # print("lm_list.shapelm_list.shape", lm_list.shape)
    results = model.predict(lm_list)
    print("results=============:", results)
    if results[0][0] > 0.5:
        result_queue.put("UONG NUOC")
    else:
        result_queue.put("KO UONG NUOC")


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
