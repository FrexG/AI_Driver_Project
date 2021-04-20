"""
Open video files and read frames
"""
# Import Opencv
import cv2 as cv
from os import path


class ReadVideoFrame:
    video_path = None

    def __init__(self, video_path):
        self.video_path = video_path
        # If path exists
        if path.exists(self.video_path):
            self.read_frame()

    def read_frame(self):
        # read video frame and display it
        cap = cv.VideoCapture(self.video_path)

        while cap.isOpened():
            ret, frame = cap.read()

            assert ret == True, "End of frame"
            cv.imshow('frame', frame)

            # 16 miliseconds per frame (60fps)

            if cv.waitKey(16) == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
