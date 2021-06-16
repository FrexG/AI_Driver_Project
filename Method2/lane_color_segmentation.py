import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class SegmentLane:
    MIN_PIXELS_PER_BIN = 1000
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
        # self.operationROI(frame)

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

        p1 = (200, height)
        p2 = (1250, height)
        p3 = (700, 450)
        p4 = (550, 450)

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
        p1 = [550, 450]
        p2 = [700, 450]
        p3 = [200, self.HEIGHT - 50]
        p4 = [1250, self.HEIGHT - 50]
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

    def inversePerspective(self, frame):
        # Define the values for the three
        # points of the polygon
        # depends on the camera and video setting
        p1 = [550, 450]
        p2 = [700, 450]
        p3 = [200, self.HEIGHT - 50]
        p4 = [1250, self.HEIGHT - 50]
        # points to be transformed
        sourcePoint = np.float32([[0, 0], [self.WIDTH, 0],
                                  [0, self.HEIGHT], [self.WIDTH, self.HEIGHT]])
        transformationPoints = np.float32([p1, p2, p3, p4])
        # get the transformation matrix

        transformMatrix = cv.getPerspectiveTransform(
            sourcePoint, transformationPoints)

        # Warp perspective
        wrapped = cv.warpPerspective(
            frame, transformMatrix, (self.WIDTH, self.HEIGHT))

        return wrapped

    def histogramPeakFinder(self, binaryFrame):
        # histogram of binary image along the x-axis

        hist = np.sum(
            binaryFrame, axis=0)
        midpoint = hist.shape[0] // 2

        leftpeak = np.argmax(hist[:midpoint])
        righpeak = np.argmax(hist[midpoint:]) + midpoint

        self.LEFT_PEAK = leftpeak
        self.RIGHT_PEAK = righpeak
        self.frame = self.slidingWindow(binaryFrame)

    def slidingWindow(self, binaryFrame):

        out = np.dstack((binaryFrame, binaryFrame, binaryFrame))
        # Window parameters #
        # window size
        numberOfWindows = 9
        # window is 10 percent of frame width
        windowWidth = 150
        windowHeight = int(binaryFrame.shape[0] / numberOfWindows)

        # starting x position of the left and right windows
        startLeftX = self.LEFT_PEAK
        startRightX = self.RIGHT_PEAK
        # Empty array to hold the indices of the left and right
        # lane pixels

        all_non_zero_x = np.nonzero(binaryFrame)[1]
        all_non_zero_y = np.nonzero(binaryFrame)[0]

        left_lane_indices = []
        right_lane_indices = []

        # draw the windows
        for i in range(numberOfWindows):
            # Current x and y positions
            y1Pos = self.HEIGHT - windowHeight * (i+1)
            y2Pos = self.HEIGHT - windowHeight * i

            left_x1Pos = startLeftX - windowWidth
            left_x2Pos = startLeftX + windowWidth

            right_x1Pos = startRightX - windowWidth
            right_x2Pos = startRightX + windowWidth

            # draw rectangle
            cv.rectangle(out, (left_x1Pos, y1Pos),
                         (left_x2Pos, y2Pos), (255, 0, 255), 3)
            cv.rectangle(out, (right_x1Pos, y1Pos),
                         (right_x2Pos, y2Pos), (0, 0, 255), 3)

            nonzero_left_window = ((all_non_zero_x < left_x2Pos) & (all_non_zero_x > left_x1Pos) &
                                   (all_non_zero_y < y2Pos) & (all_non_zero_y > y1Pos)).nonzero()[0]

            nonzero_right_window = ((all_non_zero_x < right_x2Pos) & (all_non_zero_x > right_x1Pos) &
                                    (all_non_zero_y < y2Pos) & (all_non_zero_y > y1Pos)).nonzero()[0]

            leftPixels = len(nonzero_left_window)
            rightPixels = len(nonzero_right_window)

            if leftPixels > self.MIN_PIXELS_PER_BIN:
                startLeftX = int(np.mean(all_non_zero_x[nonzero_left_window]))
                left_lane_indices.append(nonzero_left_window)
            if rightPixels > self.MIN_PIXELS_PER_BIN:
                startRightX = int(
                    np.mean(all_non_zero_x[nonzero_right_window]))
                right_lane_indices.append(nonzero_right_window)

        # Concatenate indices
        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

        # get left and right lane pixel postions
        leftX = all_non_zero_x[left_lane_indices]
        leftY = all_non_zero_y[left_lane_indices]

        rightX = all_non_zero_x[right_lane_indices]
        rightY = all_non_zero_y[right_lane_indices]

        print(rightX)
        # fit a second(third) degree polynomial to the pixel positions

        leftFit = np.polyfit(leftX, leftY, 2)

        rightFit = np.polyfit(rightX, rightY, 2)

        draw_points_left = (np.asarray([leftX, leftY]).T).astype(np.int32)
        draw_points_right = (np.asarray([rightX, rightY]).T).astype(np.int32)
        cv.polylines(out, [draw_points_left], False, (0, 0, 255), 5)
        cv.polylines(out, [draw_points_right], False, (0, 255, 0), 5)

        return cv.addWeighted(self.frame, 1, self.inversePerspective(out), 0.7, 0)
