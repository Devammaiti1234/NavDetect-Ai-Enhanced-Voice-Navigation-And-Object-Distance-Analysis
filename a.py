import cv2
import torch
import pyttsx3
import numpy as np
import math

# Initialize camera
cap = cv2.VideoCapture(0)

# Load a larger YOLOv5 model for better accuracy (you can try 'yolov5l' or 'yolov5x')
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # Using 'yolov5l' for better accuracy

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Camera intrinsic parameters (example, adjust based on your camera calibration)
focal_length = 800  # Focal length (in pixels)
known_object_width = 0.5  # Actual width of the object in meters (for example, a human's average width)

# Load class labels for YOLOv5 (COCO dataset labels)
class_labels = model.names  # Returns the object names (e.g., person, dog, car)

# Function to convert pixel distance to real-world distance
def calculate_distance(focal_length, known_width, pixel_width):
    return (focal_length * known_width) / pixel_width

# Function to calculate angle based on object's position
def calculate_angle(focal_length, object_center_x, frame_center_x, object_width):
    # The angle calculation based on object's position
    angle = math.atan((object_center_x - frame_center_x) / focal_length)
    
    # A basic adjustment for object width
    if object_width > 100:  # If the object is large in the frame, we may need to adjust angle
        angle += 0.1
    return angle

# Function to speak the calculated information
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Perform object detection on the real-time frame
    
    # Filter out low-confidence detections (lowered threshold to 0.3 for better detection)
    results = results.xywh[0]  # Results in xywh format (center_x, center_y, width, height)
    filtered_results = [det for det in results if det[4] > 0.3]  # Confidence threshold > 30%

    # Get the frame dimensions and center
    height, width, _ = frame.shape
    frame_center_x = width // 2  # Horizontal center of the frame

    # Process each detected object
    for det in filtered_results:
        x1, y1, x2, y2, confidence, class_idx = det[0], det[1], det[2], det[3], det[4], int(det[5])  # Bounding box, confidence, and class index

        # Get the object name from the class label
        object_name = class_labels[class_idx]

        # Calculate the pixel width of the object (in the bounding box)
        object_pixel_width = x2 - x1

        # Calculate the distance (in meters) to the object using the known width and focal length
        distance = calculate_distance(focal_length, known_object_width, object_pixel_width)

        # Calculate the center of the object
        object_center_x = (x1 + x2) / 2

        # Calculate the angle of the object relative to the camera's center
        angle = calculate_angle(focal_length, object_center_x, frame_center_x, object_pixel_width)

        # Convert angle from radians to degrees
        angle_deg = math.degrees(angle)

        # Provide audio feedback
        speak(f"{object_name} detected at {distance:.2f} meters, angle {angle_deg:.2f} degrees")

        # Draw the bounding box and label on the frame
        label = f"{object_name} ({confidence:.2f})"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes and labels
    cv2.imshow("Real-Time Object Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
