import cv2 as cv
import numpy as np


class SegmentLane:
    ROI_MASK = None

    def __init__(self, frame):
        self.frame = frame
        self.HEIGHT = frame.shape[0]
        self.WIDTH = frame.shape[1]
        # self.convert2HSV(frame)
        self.convert2YCRCB(frame)

    def getFrame(self):
        return self.frame

    def convert2HSV(self, frame):
        hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HLS)

        self.transformPerspective(hsvFrame)

    def convert2YCRCB(self, frame):

        ycrcbFrame = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)

        binary = self.otsuThreshold(
            self.transformPerspective(ycrcbFrame))

        self.frame = binary

    def otsuThreshold(self, colorFrame):
        ret, thresh = cv.threshold(
            colorFrame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return thresh

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
        p2 = (1050, height)

        p3 = (700, height/2)
        p4 = (450, height/2)

        roi = np.array([[p1, p2, p3, p4]], dtype=np.int32)

        cv.fillPoly(maskCanvas, roi, 255)

        self.ROI_MASK = maskCanvas

        self.overlayMask2Frame()

    def overlayMask2Frame(self):
        self.frame = cv.bitwise_and(self.frame, self.frame, mask=self.ROI_MASK)

    # adaptive ROI for feature experiments
    def transformPerspective(self, frame):
        # Define the values for the three
        # points of the polygon
        # depends on the camera and video setting
        p1 = [450, self.HEIGHT/2.2]
        p2 = [700, self.HEIGHT/2.2]
        p3 = [250, self.HEIGHT]
        p4 = [1050, self.HEIGHT]
        # points to be transformed
        sourcePoint = np.float32([p1, p2, p3, p4])
        transformationPoints = np.float32([[0, 0], [self.WIDTH, 0],
                                           [0, self.HEIGHT], [self.WIDTH, self.HEIGHT]])
        # get the transformation matrix

        transformMatrix = cv.getPerspectiveTransform(
            sourcePoint, transformationPoints)

        # Wrap perspective
        wrapped = cv.warpPerspective(
            frame, transformMatrix, (self.WIDTH, self.HEIGHT))

        return wrapped[:, :, 0]
