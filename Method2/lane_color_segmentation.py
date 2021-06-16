import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class SegmentLane:
    MIN_PIXELS_PER_BIN = 50
    ROI_MASK = None
    # left and right peaks
    LEFT_PEAK = None
    RIGHT_PEAK = None

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

        warped = self.transformPerspective(ycrcbFrame)
        binary = self.otsuThreshold(warped)

        self.histogramPeakFinder(binary)

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

        # Warp perspective
        wrapped = cv.warpPerspective(
            frame, transformMatrix, (self.WIDTH, self.HEIGHT))

        return wrapped[:, :, 0]

    def histogramPeakFinder(self, binaryFrame):
        # histogram of binary image along the x-axis
        # c = np.apply_along_axis(lambda a: np.histogram(
        #    a, bins=2)[0], axis=0, arr=binaryFrame[binaryFrame.shape[0] // 2:, :])
        c = np.sum(
            binaryFrame, axis=0)
        midpoint = c.shape[0] // 2

        leftpeak = np.argmax(c[:midpoint])
        righpeak = np.argmax(c[midpoint:]) + midpoint

        self.LEFT_PEAK = leftpeak
        self.RIGHT_PEAK = righpeak
        self.frame = self.slidingWindow(binaryFrame)

    def slidingWindow(self, binaryFrame):

        out = np.dstack((binaryFrame, binaryFrame, binaryFrame))
        # Window parameters #
        # window size
        numberOfWindows = 4
        # window is 10 percent of frame width
        windowWidth = int(binaryFrame.shape[1] * 0.1)
        windowHeight = int(binaryFrame.shape[0] / numberOfWindows)

        # starting x position of the windows
        startLeftX = self.LEFT_PEAK
        startRightX = self.RIGHT_PEAK

        # draw the windows
        for i in range(numberOfWindows):
            y1Pos = self.HEIGHT - windowHeight * (i+1)
            y2Pos = self.HEIGHT - windowHeight * i

            left_x1Pos = startLeftX - windowWidth
            left_x2Pos = startLeftX + windowWidth

            right_x1Pos = startRightX - windowWidth
            right_x2Pos = startRightX + windowWidth

            cv.rectangle(out, (left_x1Pos, y1Pos),
                         (left_x2Pos, y2Pos), (0, 255, 0), 2)
            cv.rectangle(out, (right_x1Pos, y1Pos),
                         (right_x2Pos, y2Pos), (255, 0, 0), 2)

            all_non_zero_x = np.nonzero(binaryFrame)[1]
            all_non_zero_y = np.nonzero(binaryFrame)[0]

            nonzero_left_window = ((all_non_zero_x < left_x2Pos) & (all_non_zero_x > left_x1Pos) &
                                   (all_non_zero_y < y2Pos) & (all_non_zero_y > y1Pos)).nonzero()[0]

            nonzero_right_window = ((all_non_zero_x < right_x2Pos) & (all_non_zero_x > right_x1Pos) &
                                    (all_non_zero_y < y2Pos) & (all_non_zero_y > y1Pos)).nonzero()[0]

            leftPixels = len(nonzero_left_window)
            rightPixels = len(nonzero_right_window)

            if leftPixels > self.MIN_PIXELS_PER_BIN:
                startLeftX = int(np.mean(all_non_zero_x[nonzero_left_window]))
            if rightPixels > self.MIN_PIXELS_PER_BIN:
                startRightX = int(
                    np.mean(all_non_zero_x[nonzero_right_window]))

        return out
