import cv2
import pandas as pd
import torch
import math
import os
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
import numpy as np


parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

EDGES = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
}

# config tf
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures["serving_default"]


def make_data_from_video():
    data_to_write = []

    cap = cv2.VideoCapture(parent_dir + "/data/noaction.mp4")

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            print("Error: Unable to read frame.")
            break

        _img = img.copy()
        _img = tf.image.resize_with_pad(tf.expand_dims(_img, axis=0), 384, 640)
        input_img = tf.cast(_img, dtype=tf.int32)

        # Detection section
        results = movenet(input_img)
        # print("results=======", results["output_0"])
        keypoints_with_scores = (
            results["output_0"].numpy()[:, :, :51].reshape((6, 17, 3))
        )

        # Render keypoints
        loop_through_people(img, keypoints_with_scores, EDGES, 0.1)

        cv2.imshow("image", img)
        cv2.waitKey(1)

    # save to txt file
    # print("Writing file...")
    # file_name = "noaction"
    # df = pd.DataFrame(data_to_write)
    # df.to_csv("files/" + file_name + ".txt")

    cap.release()
    cv2.destroyAllWindows()


def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


def draw_keypoints(frame, keypoints, confidence_threshold):
    print("keypointskeypointskeypoints", keypoints)
    y, x, c = frame.shape

    print("==============", y, x, c)

    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)


if __name__ == "__main__":
    make_data_from_video()
