# from video_processor import process_video
from video_runner import process_video
from ultralytics import YOLO

if __name__ == "__main__":

    try:
        model = YOLO("yolov8s.pt")
        process_video("test_videos/video.mp4", model)

    except Exception as e:
        print(f"Error occurred: {e}")





# prints the list of built-in classes in the yolo model
#print(model.names)