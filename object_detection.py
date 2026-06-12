import cv2
import torch
import pyttsx3
import numpy as np
from ultralytics import YOLO
import threading
import time
from functools import partial


# Load YOLOv8 model (Most powerful version)
model = YOLO("yolov8x.pt")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Text-to-speech (Runs in a separate thread to prevent lag)
engine = pyttsx3.init()
engine.setProperty('rate', 170)  # Faster speech
engine.setProperty('volume', 1.0)

# def speak(text):
#     threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()), daemon=True).start()
def speak_function(text):
    engine.say(text)
    engine.runAndWait()

def speak(text):
    threading.Thread(target=partial(speak_function, text), daemon=True).start()

# Mobile Camera Stream URL (Update this with your actual mobile IP)
MOBILE_CAMERA_URL = "http://192.168.0.100:8080/video"  # Change this to your IP

# Use threaded video capture for higher FPS
class VideoCapture:
    def __init__(self, source=MOBILE_CAMERA_URL):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 90)  # Higher FPS
        self.ret, self.frame = self.cap.read()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

# Start video capture from mobile camera
cap = VideoCapture()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    start_time = time.time()  # FPS Timer

    # Run YOLOv8 object detection
    results = model(frame, conf=0.6)  

    detected_objects = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        names = result.names  # Class names

        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score < 0.6:
                continue  

            x1, y1, x2, y2 = map(int, box)
            label = names[int(result.boxes.cls[i])]

            # Ensure proper bounding box placement
            x1, y1 = max(0, x1), max(0, y1)

            # **Reduce the detection box size slightly**
            box_padding = 10
            x1, y1 = x1 + box_padding, y1 + box_padding
            x2, y2 = x2 - box_padding, y2 - box_padding

            # Draw bounding box with **smaller size**
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # **Thinner bounding box**
            cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # **Smaller font**

            detected_objects.append(f"{label}")

    # Speak detected objects
    if detected_objects:
        speak(", ".join(detected_objects))

    # Show FPS on screen
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow('Mobile Camera Object Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
