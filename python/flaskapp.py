# Install Flask on your system by writing
#!pip install Flask
# Import all the required libraries
# Importing Flask
# render_template--> To render any html file, template

from flask import Flask, Response, jsonify, request
from PIL import Image
import numpy as np

# Required to run the YOLOv8 model
import cv2

# YOLO_Video is the python file which contains the code for our object detection model
# Video Detection is the Function which performs Object Detection on Input Video
from YOLO_Video import video_detection, handleDetect

app = Flask(__name__)

app.config["SECRET_KEY"] = "muhammadmoin"
# Generate_frames function takes path of input video file and  gives us the output with bounding boxes
# around detected objects


# Now we will display the output video with detection
def generate_frames(path_x=""):
    # yolo_output variable stores the output for each detection
    # the output with bounding box around detected objects

    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode(".jpg", detection_)
        # Any Flask application requires the encoded image to be converted into bytes
        # We will display the individual frames using Yield keyword,
        # we will loop over all individual frames and display them as video
        # When we want the individual frames to be replaced by the subsequent frames the Content-Type, or Mini-Type
        # will be used
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video")
def video():
    return Response(
        generate_frames(path_x="./data/wine1.mp4"),
        # generate_frames(path_x="https://cdn.shinobi.video/videos/theif4.mp4"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/image")
def detectImage():
    return Response(
        generate_frames(path_x="pose.jpg"),
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
