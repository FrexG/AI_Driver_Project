import cv2 as cv
import numpy as np


class LaneDetector:
    road_frame = None
    roi_params = []

    def __init__(self, label):
        if label:
            self.roi_params = [150, 1000, (650, 520)]
        else:
            self.roi_params = [100, 1150, (650, 400)]

    def detect_lane(self, lane_frame, label):

        self.road_frame = lane_frame

        # as a test convert the frame to gray scale
        gray = cv.cvtColor(self.road_frame, cv.COLOR_RGB2GRAY)
        # Gaussian blur
        blur = cv.GaussianBlur(gray, (5, 5), 0)

        canny = self.canny(blur)

        mask = self.create_roi_mask(canny)

        masked_blur = cv.bitwise_and(canny, canny, mask=mask)

        lines_img = self.display_hough_lines(self.road_frame, masked_blur)

        added_img = cv.addWeighted(self.road_frame, 0.8, lines_img, 1, 1)

        return added_img

    def create_roi_mask(self, image):
        roi_mask = np.zeros_like(image)
        image_height = image.shape[0]
        roi_triangle = np.array([
            [(self.roi_params[0], image_height),
             (self.roi_params[1], image_height), self.roi_params[2]]
        ])

        cv.fillPoly(roi_mask, roi_triangle, 255)

        return roi_mask

    def canny(self, gray):
        canny = cv.Canny(gray, 50, 150)
        return canny

    def calculate_hough_lines(self, roi):
        # 2 pixel spatial percission
        # 1 degree(radians) percission
        hough_lines = cv.HoughLinesP(roi, 2, np.pi/180, 70, np.array([]),
                                     minLineLength=40, maxLineGap=5)
        return hough_lines

    def display_hough_lines(self, img, roi):
        lines = self.calculate_hough_lines(roi)
        lines_canvas = np.zeros_like(img)
        if lines is not None:
            print(f'lines at{lines}')
            averaged_lines = self.hough_averages(lines)
            left, right = averaged_lines
            x1, y1, x2, y2 = left
            cv.line(lines_canvas, (x1, y1), (x2, y2), (19, 255, 54), 10)
            x1, y1, x2, y2 = right
            cv.line(lines_canvas, (x1, y1), (x2, y2), (19, 255, 54), 10)

        return lines_canvas

    def hough_averages(self, lines):
        left_lane = []
        right_lane = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)

            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_lane.append((slope, intercept))
            else:
                right_lane.append((slope, intercept))

            left_lane_avg = np.average(left_lane, 0)

            right_lane_avg = np.average(right_lane, 0)

        left_line = self.makeLine(left_lane_avg)

        right_line = self.makeLine(right_lane_avg)

        return (left_line, right_line)

    def makeLine(self, lane):
        slope, intercept = lane
        y1 = int(self.road_frame.shape[0])
        y2 = int(y1 * 1/2)

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return (x1, y1, x2, y2)
