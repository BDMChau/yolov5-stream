import cv2
import pandas as pd
from ultralytics import YOLO
import torch
import math
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# config YOLO
modelPersonPose = YOLO("./weights/yolov8s-pose.pt").to(device)


def make_data_from_video():
    data_to_write = []

    cap = cv2.VideoCapture(parent_dir + "/data/only_drinking.mp4")

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            print("Error: Unable to read frame.")
            break

        modelPersonPoseResults = modelPersonPose(
            img,
            stream=True,
        )

        for r in modelPersonPoseResults:
            if r.boxes is None:
                continue

            names = r.names
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                class_id = int(box.cls[0])
                class_name = names[class_id]
                label = f"{class_name}_{conf}"
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

            # POSE
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
            result_timestep = []
            for i, keypoint_tensor in enumerate(keypoints_xyn):
                x, y = (
                    keypoint_tensor[0].item(),
                    keypoint_tensor[1].item(),
                )

                result_timestep.append(x)
                result_timestep.append(y)

            print(result_timestep)
            data_to_write.append(result_timestep)

        cv2.imshow("image", img)
        cv2.waitKey(1)

    # save to txt file
    print("Writing file...")
    file_name = "theft01"
    df = pd.DataFrame(data_to_write)
    df.to_csv("files/" + file_name + ".txt")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    make_data_from_video()
