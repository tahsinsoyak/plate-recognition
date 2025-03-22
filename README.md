# Vehicle License Plate Detection and Tracking System

A real-time video processing system that detects vehicles, tracks their movement, and recognizes license plates using computer vision and deep learning. The system provides high-accuracy vehicle tracking and license plate recognition with frame-by-frame visualization capabilities.

This project combines YOLO object detection, SORT tracking algorithm, and optical character recognition (OCR) to create a comprehensive vehicle monitoring solution. It processes video streams to detect vehicles (cars, motorcycles, buses, and trucks), track their movement across frames, and extract license plate information. The system handles missing data through interpolation and provides visualization tools for reviewing the results.

Key features include:
- Real-time vehicle detection and tracking using YOLOv8 and SORT algorithm
- License plate detection and OCR-based text extraction
- Automatic interpolation of missing tracking data
- Visual output with bounding boxes and license plate overlays
- CSV export of detection and tracking results

## Project Structure
```
plate-recognition/
├── src/
│   ├── detection/          # Detection modules
│   ├── tracking/           # Tracking modules
│   ├── visualization/      # Visualization scripts
│   ├── utils/              # Utility functions
│   ├── main.py             # Main entry point
│   ├── config.py           # Configuration file
├── models/                 # Pre-trained models
├── data/                   # Input/output data
```

## Usage Instructions
### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- Video file for processing

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install -r requirements.txt

# Download required YOLO models
# - yolov8n.pt (COCO-trained model)
# - license_plate_detector.pt (Custom license plate detection model)
```

### Quick Start
1. Place your input video file in the project directory
2. Run the detection and tracking:
```python
python main.py
```
3. View the results:
```python
python visualize.py
```

### More Detailed Examples
1. Process a video with custom tracking parameters:
```python
from sort import Sort
from main import process_video

# Initialize tracker with custom parameters
tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.4)

# Process video
process_video('path/to/video.mp4', tracker)
```

2. Interpolate missing tracking data:
```python
python add_missing_data.py
```

### Troubleshooting
Common issues and solutions:

1. CUDA Out of Memory
- Problem: GPU memory overflow during detection
- Solution: Reduce batch size or video resolution
```python
# In main.py, add frame resizing
frame = cv2.resize(frame, (1280, 720))
```

2. Missing License Plate Detections
- Problem: Poor license plate recognition in challenging conditions
- Solution: Adjust detection confidence threshold
```python
# In main.py, modify detection threshold
license_plates = license_plate_detector(frame, conf=0.25)[0]
```

3. Debug Mode
To enable verbose logging:
```python
# Add to beginning of main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Data Flow
The system processes video frames through a pipeline of detection, tracking, and recognition stages.

```ascii
Video Input → Vehicle Detection → Vehicle Tracking → License Plate Detection → OCR → CSV Output
     ↓              ↓                   ↓                    ↓               ↓         ↓
[Frame] → [YOLO] → [SORT] → [License Detector] → [EasyOCR] → [Results] → [CSV/Video]
```

Key component interactions:
1. Main loop reads video frames sequentially
2. YOLO model detects vehicles in each frame
3. SORT tracker maintains vehicle IDs across frames
4. License plate detector identifies plates within vehicle regions
5. OCR extracts text from detected license plates
6. Results are stored in a frame-indexed dictionary
7. Final data is exported to CSV and visualized in output video
8. Interpolation fills gaps in tracking data