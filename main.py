from yolo_model import load_model
from video_processor import process_video

# loading the yolo model
model = load_model()

# testing analysis on a test image
#results = model("test.jpg")
#results[0].show()

#testing with video
process_video("video.mp4", model)

# prints the list of built-in classes in the yolo model
#print(model.names)