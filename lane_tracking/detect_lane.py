import cv2 as cv
import numpy as np


class LaneDetector:
    road_frame = None

    def detect_lane(self, lane_frame):
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
            [(250, image_height), (1050, image_height), (620, 500)]
        ])

        cv.fillPoly(roi_mask, roi_triangle, 255)

        return roi_mask

    def canny(self, gray):
        canny = cv.Canny(gray, 50, 150)
        return canny

    def calculate_hough_lines(self, roi):
        # 2 pixel spatial percission
        # 1 degree(radians) percission
        hough_lines = cv.HoughLinesP(roi, 2, np.pi/180, 100, np.array([]),
                                     minLineLength=60, maxLineGap=5)
        return hough_lines

    def display_hough_lines(self, img, roi):
        lines = self.calculate_hough_lines(roi)
        lines_canvas = np.zeros_like(img)

        if lines is not None:
            for line in lines:

                x1, y1, x2, y2 = line.reshape(4)
                cv.line(lines_canvas, (x1, y1), (x2, y2), (0, 255, 0), 10)

        return lines_canvas
