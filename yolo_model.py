from ultralytics import YOLO

def load_model():
    model = YOLO("yolov8n.pt")  # מודל מוכן קטן ומהיר
    return model