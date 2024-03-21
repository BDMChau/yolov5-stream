from flask import Flask, Response, jsonify, request
from PIL import Image
import numpy as np

import cv2

from YOLO_Video import video_detection, handleDetect

app = Flask(__name__)


def generate_frames(path_x=""):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode(".jpg", detection_)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video")
def video():
    return Response(
        generate_frames(path_x="./data/drinking.mp4"),
        # generate_frames(path_x="https://cdn.shinobi.video/videos/theif4.mp4"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream")
def stream():
    # url = "rtsp://raptor:Raptor123!@192.168.100.36:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
    url = "http://192.168.100.252:8989/get-stream/ZGV2LXJhcHRvci1haQ--/cnRzcDovL3JhcHRvcjpSYXB0b3IxMjMhQDE5Mi4xNjguMTAwLjM2OjU1NC9jYW0vcmVhbG1vbml0b3I_Y2hhbm5lbD0xJnN1YnR5cGU9MCZ1bmljYXN0PXRydWUmcHJvdG89T252aWY-"
    return Response(
        generate_frames(path_x=url),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        print("fileBufferfileBuffer", file)
        image = Image.open(file.stream).convert("RGB")
        img = np.array(image)

        result = handleDetect(img)

        print("result", result)
        return jsonify(result)
    except Exception as e:
        return jsonify({})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3003, debug=True)
