import argparse
from flask import Flask, request, jsonify
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh, check_img_size
from utils.torch_utils import select_device, smart_inference_mode
from PIL import Image
import numpy as np
import logging
from werkzeug import serving
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8989, help='port number')
parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model path')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--imgszw', type=int, default=640, help='inference width (pixels)')
parser.add_argument('--imgszh', type=int, default=480, help='inference height (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.6, help='confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
parser.add_argument('--classes-file', type=str, default='weights/classes.txt', help='Path to classes.txt file')
parser.add_argument('--datayaml', type=str, default='data/coco128.yaml', help='dataset.yaml path')
args = parser.parse_args()

app = Flask(__name__)

print(torch.cuda.is_available())

# Load class names
with open(args.classes_file, 'r') as f:
    names = [line.strip() for line in f.readlines()]

# Load the model
device = select_device(args.device)
model = DetectMultiBackend(args.weights, device=device, dnn=False, data=args.datayaml, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = (args.imgszw, args.imgszh)
imgsz = check_img_size(imgsz, s=stride)

@app.route('/detect', methods=['POST'])
@smart_inference_mode()
def detect():

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image = Image.open(file.stream).convert('RGB')
    img = np.array(image)

    # Convert image to PyTorch tensor and permute dimensions to [C, H, W]
    im = torch.from_numpy(img).permute(2, 0, 1).to(device)  # Permute dimensions to [C, H, W]
    im = im.half() if model.fp16 else im.float()  # Convert image to half precision float if model uses FP16, else use FP32
    im /= 255.0  # Normalize image from 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # Add a batch dimension
        

    # Inference
    if hasattr(model, 'xml') and model.xml and im.shape[0] > 1:  # This condition is model specific and might not be relevant for all YOLOv5 models

        pred = None
        ims = torch.chunk(im, im.shape[0], 0)  # Split the batch into individual images
        for image in ims:
            if pred is None:
                pred = model(image, augment=False, visualize=False).unsqueeze(0)
            else:
                pred = torch.cat((pred, model(image, augment=False, visualize=False).unsqueeze(0)), dim=0)
        pred = [pred, None]
    else:
        pred = model(im, augment=False, visualize=False)

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, 1000)
    # pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, args.classes, args.agnostic_nms,max_det=args.max_det)


    # Process detections and format the response
    detections = []
    for i, det in enumerate(pred):  # Iterate through images
        print(det)
        if len(det):
            # Rescale boxes from img_size to original size
            det[:, :4] = scale_boxes((imgsz[1], imgsz[0]), det[:, :4], img.shape).round()

            for *xyxy, conf, cls in reversed(det):
                # Convert bounding box format
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                x1, y1, w, h = xywh
                x = x1 - (w / 2)  # Calculate top left x
                y = y1 - (h / 2)  # Calculate top left y

                detections.append({
                    'tag': names[int(cls)],
                    'confidence': float(conf),
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                })
    
    return jsonify(detections)

if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='127.0.0.1', port=args.port)  # Start the Flask server

# python3 yolov5/yolov5_flask_server.py --port 8989 --weights ./weights/yolov5-vip.pt

# python3 ./detect.py --weights weights/yolov5s.pt --source http://192.168.100.252:8989/get-stream/cWEtc2l0ZQ--/cnRzcDovL3JhcHRvcjpSYXB0b3IxMjMhQDE5Mi4xNjguMTAwLjEzMjo1NTQvY2FtL3JlYWxtb25pdG9yP2NoYW5uZWw9MSZzdWJ0eXBlPTAmdW5pY2FzdD10cnVlJnByb3RvPU9udmlm

# python3 ./detect.py --weights weights/yolov5s.pt --source http://192.168.100.252:8989/get-stream/cWEtc2l0ZQ--/cnRzcDovL3JhcHRvcjpSYXB0b3IxMjMhQDE5Mi4xNjguMTAwLjM2OjU1NC9jYW0vcmVhbG1vbml0b3I_Y2hhbm5lbD0xJnN1YnR5cGU9MCZ1bmljYXN0PXRydWUmcHJvdG89T252aWY-