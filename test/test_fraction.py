from fractions import Fraction

class VideoFrame:
    def __init__(self):
        self.time_base = None

video_frame = VideoFrame()
fps = 15
video_frame.time_base = Fraction(1, fps)
print(video_frame.time_base)