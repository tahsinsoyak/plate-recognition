from ultralytics import YOLO
import config

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame, classes=None):
        results = self.model(frame)[0]
        detections = []
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if classes is None or int(class_id) in classes:
                detections.append([x1, y1, x2, y2, score, int(class_id)])
        return detections
