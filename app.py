from flask import Flask, render_template, Response
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import threading
import time

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("yolov8x.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Capture from mobile camera (IP Webcam App Required)
MOBILE_CAM_URL = "http://your_mobile_ip:8080/video"
cap = cv2.VideoCapture(MOBILE_CAM_URL)

KNOWN_OBJECT_WIDTHS = {
    "person": 0.45, "car": 1.8, "bottle": 0.07, "chair": 0.5, "cup": 0.08,
    "laptop": 0.35, "book": 0.3, "tv": 1.0, "cell phone": 0.15, "id card": 0.085
}
FOCAL_LENGTH = 850

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        results = model(frame, conf=0.6)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            names = result.names
            
            for i, (box, score) in enumerate(zip(boxes, scores)):
                if score < 0.6:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                label = names[int(result.boxes.cls[i])]
                object_width_pixels = x2 - x1
                
                if label in KNOWN_OBJECT_WIDTHS:
                    real_width = KNOWN_OBJECT_WIDTHS[label]
                    distance = (real_width * FOCAL_LENGTH) / (object_width_pixels + 1e-6)
                    distance_text = f"{label}: {distance:.2f} meters"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, distance_text, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
