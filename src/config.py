import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# Model files
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
LICENSE_PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "license_plate_detector.pt")

# Verify model files exist
if not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model file not found at {YOLO_MODEL_PATH}. Please ensure the file exists in the 'models' directory."
    )
if not os.path.exists(LICENSE_PLATE_MODEL_PATH):
    raise FileNotFoundError(
        f"License plate detector model file not found at {LICENSE_PLATE_MODEL_PATH}. "
        f"Please ensure the file exists in the 'models' directory or download it from the appropriate source."
    )

# Input/Output files
VIDEO_PATH = os.path.join(INPUT_DIR, "video.mp4")
CSV_PATH = os.path.join(OUTPUT_DIR, "test.csv")
INTERPOLATED_CSV_PATH = os.path.join(OUTPUT_DIR, "test_interpolated.csv")
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "out.mp4")

# Detection parameters
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
