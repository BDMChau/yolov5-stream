import cv2
import pandas as pd
import torch
import keras
import os
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
import numpy as np
from utils import normalize_pose_landmarks, landmarks_to_embedding

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

# https://www.kaggle.com/models?query=movenet&tfhub-redirect=true
model = hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1")
movenet = model.signatures["serving_default"]


def make_data_from_video():
    data_to_write = []

    cap = cv2.VideoCapture(parent_dir + "/data/drinking-man.mp4")

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            print("Error: Unable to read frame.")
            break

        imgToResize = img.copy()
        imgToResize = resize_and_pad_image(imgToResize)
        resize_height, resize_width = imgToResize.shape[:2]
        
        imgToDetect = tf.cast(tf.image.resize_with_pad(tf.expand_dims(img, axis=0), resize_height, resize_width), dtype=tf.int32)
 
        # Detection section, output is a [1, 6, 56] tensor.
        results = movenet(imgToDetect)
        results = results["output_0"];
        keypoints_with_scores = (
            results.numpy()[:, :, :51].reshape((6, 17, 3))
        )
        
           
        # embeddedLandmarks = landmarks_to_embedding(normalizedLandmarks)
        # print("embeddedLandmarks=======", embeddedLandmarks)
        
        # Normalize landmarks 2D
        # normalizedLandmarks = normalize_pose_landmarks(keypoints_with_scores[:, :, :2])
        # print("normalizeLandmarks=======", normalizedLandmarks)


        # Render keypoints
        loop_through_people(img, keypoints_with_scores, EDGES, 0.2)

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
    for keypoints in keypoints_with_scores:
        print("keypoints: ", keypoints)
        draw_connections(frame, keypoints, edges, confidence_threshold)
        draw_keypoints(frame, keypoints, confidence_threshold)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x = frame.shape[:2]

    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 5, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
def resize_and_pad_image(image, target_size=(256, 256)):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    
    if h > w:
        new_h = target_size[0]
        new_w = int(new_h * aspect_ratio)
    else:
        new_w = target_size[0]
        new_h = int(new_w / aspect_ratio)
    
    new_image = cv2.resize(image, (new_w, new_h))
    
    new_image = cv2.copyMakeBorder(new_image, 0, (32 - new_h % 32) % 32, 0, (32 - new_w % 32) % 32, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return new_image


if __name__ == "__main__":
    make_data_from_video()
