from lane_tracking.open_video import ReadVideoFrame

video_1 = {"path": "/home/frexg/Downloads/Lane Detection Video.mp4", "label": 1}
video_2 = {"path": "/home/frexg/Downloads/test2.mp4", "label": 0}

if __name__ == "__main__":
    ReadVideoFrame(video_1)
