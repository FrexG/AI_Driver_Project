"""
Open video files and read frames
"""
# Import Opencv
import cv2 as cv
from .detect_lane import LaneDetector
from os import path


class ReadVideoFrame:
    video_path = None

    def __init__(self, video_path):
        self.video_path = video_path
        # If path exists
        if path.exists(self.video_path):
            self.read_frame()
        else:
            raise FileNotFoundError("File Doesn't exist")

    def read_frame(self):
        # read video frame and display it
        cap = cv.VideoCapture(self.video_path)

        # Create a LaneDetector object
        detector = LaneDetector()

        while cap.isOpened():
            ret, frame = cap.read()

            assert ret == True, "End of frame"

            gray = detector.detect_lane(frame)

            cv.imshow('frame', gray)

            # 16 miliseconds per frame (60fps)

            if cv.waitKey(16) == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
