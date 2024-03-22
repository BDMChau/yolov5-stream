from ultralytics import YOLO
import cv2
from flask import jsonify
from ultralytics.utils.ops import xyxy2xywh
import torch
import math
import yaml
import numpy as np
import threading
import tensorflow as tf
import queue

classNamesCoco = [
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

# force tensorflow use CPU
# tf.config.set_visible_devices([], "GPU")

# allow GPU memory for tf when need
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


print(f"Using device for YOLO: {device}")
modelPersonPose = YOLO("./weights/yolov8s-pose.pt").to(device)
modelObjectDetection = YOLO("./weights/yolov8x.pt").to(device)

lstm_model = tf.keras.models.load_model("./LSTM/results/lstm01.keras")

window_size = 10


def video_detection(path_x):
    time_steps_items = {}
    lstm_labels = {}

    result_queues = {}

    cap = cv2.VideoCapture(path_x)

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
                class_name = classNamesCoco[class_id]
                label = f"{ids[i]}:{class_name}_{conf}"

                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                x, y, w, h = xywh
                x = x - (w / 2)
                y = y - (h / 2)

                if (
                    True
                    # class_name
                    # != "bottle"
                    # and class_name != "person"
                    #     # and class_name != "Wine"
                    #     # and class_name != "Man"
                    #     # and class_name != "Woman"
                ):
                    continue

                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
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
            img = r.plot(kpt_line=True, kpt_radius=5)

            items_ltsm = {}
            for i, keypoints_xyn in enumerate(r.keypoints.xyn):
                keypoints_xy = r.keypoints.xy[i]

                # 17 points of body
                for j, point_xyn in enumerate(keypoints_xyn):
                    x, y = (
                        int(keypoints_xy[j][0].item()),
                        int(keypoints_xy[j][1].item()),
                    )
                    xn, yn = (
                        point_xyn[0].item(),
                        point_xyn[1].item(),
                    )

                    track_id = ids[i]

                    if track_id not in items_ltsm:
                        items_ltsm[track_id] = []

                    items_ltsm[track_id].append(xn)
                    items_ltsm[track_id].append(yn)

                    # cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

            for key, value in items_ltsm.items():
                if len(value) > 0:
                    if key not in time_steps_items:
                        time_steps_items[key] = []

                    time_steps_items[key].append(value)
                    items_ltsm[key] = []

            for key, value in time_steps_items.items():
                if len(value) == window_size:
                    result_queues[key] = queue.Queue()
                    lstmDetect_thread = threading.Thread(
                        target=lstmDetect,
                        args=(
                            lstm_model,
                            value,
                            result_queues[key],
                        ),
                    )
                    lstmDetect_thread.start()

                    time_steps_items[key] = []

            for key, value in result_queues.items():
                if not value.empty():
                    lstm_labels[key] = value.get()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                class_id = int(box.cls[0])
                class_name = names[class_id]
                track_id = ids[i]

                lstm_label = "Nothing"
                if track_id in lstm_labels:
                    lstm_label = lstm_labels[track_id]

                label = f"{lstm_label}"
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] + 25

                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                cv2.rectangle(img, (x1, y1 + 25), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(
                    img,
                    label,
                    (x1, y1 + 25),
                    0,
                    1,
                    [255, 255, 255],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        print("lstm_labels lstm_labelsAAAAAAAA:", lstm_labels)
        yield img


def lstmDetect(model, lm_list, result_queue):
    class_labels = ["hand swing", "PUNCH nghien", "No action"]

    threshold = 0.75

    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)

    results = model.predict(lm_list)
    maxValueIndex = np.argmax(results)

    print("PREDICT RESULTS:", results)
    if maxValueIndex < len(class_labels):
        if results[0][maxValueIndex] >= threshold:
            final_result = class_labels[maxValueIndex]
        else:
            final_result = "Different Behavior"
    else:
        final_result = "Different Behavior"

    result_queue.put(final_result)


def handleDetect(img):

    return []
