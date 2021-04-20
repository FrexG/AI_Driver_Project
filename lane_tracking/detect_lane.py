import cv2 as cv
import numpy as np


class LaneDetector:
    road_frame = None

    def detect_lane(self, lane_frame):
        self.road_frame = lane_frame

        # as a test convert the frame to gray scale
        gray = cv.cvtColor(self.road_frame, cv.COLOR_RGB2GRAY)

        canny = self.canny(gray)

        return canny

    def create_roi_mask(self, image):
        mask = np.zeros_like(image)

    def canny(self, gray):
        canny = cv.Canny(gray, 100, 255)

        return canny
