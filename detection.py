import os
import cv2
import pandas as pd
import time
from tracker import Tracker
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Resize
from ultralytics.yolov5 import Model

# Load YOLO model
model = Model('yolov5s.pt')

# Define class list
class_list = ['car']

# Initialize tracker
tracker = Tracker()

# Video parameters
frame_width = 1020
frame_height = 500
fps = 20.0
output_filename = 'output.avi'

# Line positions
red_line_y = 198
blue_line_y = 268
offset = 6

# Create folder to save frames
if not os.path.exists('detected_frames'):
    os.makedirs('detected_frames')

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Open video file
cap = cv2.VideoCapture('highway.mp4')

# Initialize variables
count = 0
down = {}
up = {}
counter_down = []
counter_up = []

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

    # Resize frame
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Convert to PIL image and apply YOLO inference
    pil_img = Image.fromarray(frame)
    img = Resize((640, 640))(pil_img)
    img = to_tensor(img).unsqueeze(0)
    results = model(img)

    # Process YOLO results
    boxes = results.xyxy[0].cpu().numpy()
    detections = [(int(box[0]), int(box[1]), int(box[2]), int(box[3])) for box in boxes if box[5] == 2]  # Filter only car detections

    # Update tracker
    bbox_id = tracker.update(detections)

    # Process tracked objects
    for bbox in bbox_id:
        x1, y1, x2, y2, id = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Calculate vehicle speed
        if red_line_y - offset <= cy <= red_line_y + offset:
            down[id] = time.time() if id not in down else down[id]
        elif blue_line_y - offset <= cy <= blue_line_y + offset:
            up[id] = time.time() if id not in up else up[id]

        if id in down and id in up:
            elapsed_time = up[id] - down[id]
            speed_kmh = 10 / elapsed_time * 3.6  # 10 meters between lines
            speed_text = f'{int(speed_kmh)} Km/h'
            cv2.putText(frame, speed_text, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Draw lines and other annotations
    cv2.line(frame, (172, red_line_y), (774, red_line_y), (0, 0, 255), 2)
    cv2.line(frame, (8, blue_line_y), (927, blue_line_y), (255, 0, 0), 2)

    # Save frame
    frame_filename = f'detected_frames/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)

    # Write frame to output video
    out.write(frame)

    # Display frame
    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()