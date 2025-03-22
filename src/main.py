import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from config import VIDEO_PATH, CSV_PATH, VEHICLE_CLASSES, YOLO_MODEL_PATH, LICENSE_PLATE_MODEL_PATH
from detection.yolo_detector import YOLODetector
from tracking.sort import Sort
from utils.data_utils import write_csv

# Initialize models and tracker
vehicle_detector = YOLODetector(YOLO_MODEL_PATH)
license_plate_detector = YOLODetector(LICENSE_PLATE_MODEL_PATH)
tracker = Sort()

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
results = []

frame_nmr = -1
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_nmr += 1

    # Detect vehicles
    vehicle_detections = vehicle_detector.detect(frame, VEHICLE_CLASSES)
    tracked_vehicles = tracker.update(np.array(vehicle_detections))

    # Detect license plates
    license_plate_detections = license_plate_detector.detect(frame)

    # Process detections
    for track in tracked_vehicles:
        x1, y1, x2, y2, track_id = track
        for lp in license_plate_detections:
            lx1, ly1, lx2, ly2, score, _ = lp
            if lx1 > x1 and ly1 > y1 and lx2 < x2 and ly2 < y2:
                results.append({
                    "frame_nmr": frame_nmr,
                    "car_id": track_id,
                    "car_bbox": f"[{x1} {y1} {x2} {y2}]",
                    "license_plate_bbox": f"[{lx1} {ly1} {lx2} {ly2}]",
                    "license_plate_bbox_score": score,
                })

# Write results to CSV
write_csv(results, CSV_PATH, ["frame_nmr", "car_id", "car_bbox", "license_plate_bbox", "license_plate_bbox_score"])
cap.release()
