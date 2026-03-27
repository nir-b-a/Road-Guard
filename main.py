from yolo_model import load_model

model = load_model()
results = model("test.jpg")
results[0].show()