import cv2 as cv
import numpy as np


class SegmentLane:
    ROI_MASK = None

    def __init__(self, frame):
        self.frame = frame
        self.operationROI(frame)

    def getFrame(self):
        return self.frame[:, :, 2]

    def convert2HSV(self, frame):
        self.frame = cv.cvtColor(frame, cv.COLOR_BGR2HLS)

    def operationROI(self, frame):
        maskCanvas = np.zeros_like(frame[:, :, 0])
        # parameters for the ROI differ based on the
        # camera type and image type

        # Image height
        height = frame.shape[0]

        # Define the values for the three
        # points of the polygon
        # depends on the camera and video setting

        p1 = (250, height)
        p2 = (1000, height)

        p3 = (600, height/4)
        p4 = (550, height/4)

        roi = np.array([[p1, p2, p3, p4]], dtype=np.int32)

        cv.fillPoly(maskCanvas, roi, 255)

        self.ROI_MASK = maskCanvas

        self.overlayMask2Frame()

        self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)

    def overlayMask2Frame(self):
        self.frame = cv.bitwise_and(self.frame, self.frame, mask=self.ROI_MASK)
