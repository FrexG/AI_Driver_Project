"""
Open video files and read frames
"""
# Import Opencv
import cv2 as cv
from .detect_lane import LaneDetector
from os import path


class ReadVideoFrame:
    video = None
    detector = None

    def __init__(self, video):
        self.video = video
        # If path exists
        if path.exists(self.video['path']):
            self.detector = LaneDetector(self.video['label'])
            self.read_frame()
        else:
            raise FileNotFoundError("File Doesn't exist")

    def read_frame(self):
        # read video frame and display it
        cap = cv.VideoCapture(self.video['path'])

        # Create a LaneDetector object

        while cap.isOpened():
            ret, frame = cap.read()

            assert ret == True, "End of frame"

            gray = self.detector.detect_lane(frame, self.video['label'])

            cv.imshow('frame', gray)

            # 16 miliseconds per frame (60fps)

            if cv.waitKey(1) == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
