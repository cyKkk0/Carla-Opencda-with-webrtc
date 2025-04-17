import os
import cv2
import time
import random
import asyncio
import fractions
import numpy as np

from av import VideoFrame
from collections import deque
from aiortc import VideoStreamTrack

VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 30  # 20fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)


class FpsCalculator:
    def __init__(self, window_size=30):
        """
        初始化帧率计算器。
        
        :param window_size: 滑动窗口的大小（单位：帧数），用于计算最近的帧率。
        """
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.fps = 0.0
        self.last_calculated_time = time.time()

    def update(self, frame_count=1):
        """
        更新帧率计算器。
        
        :param frame_count: 本次调用发送的帧数(默认为1)。
        """
        current_time = time.time()
        self.timestamps.extend([current_time] * frame_count)

        # 如果窗口已满或达到最小时间间隔，计算帧率
        if len(self.timestamps) >= self.window_size or (current_time - self.last_calculated_time) >= 0.1:
            if len(self.timestamps) < 2:
                return

            # 计算窗口内的时间跨度
            time_span = self.timestamps[-1] - self.timestamps[0]
            if time_span <= 0:
                return

            # 计算帧率
            self.fps = len(self.timestamps) / time_span
            self.last_calculated_time = current_time

    def get_fps(self):
        """
        获取当前的帧率。
        
        :return: 当前的帧率(FPS)
        """
        return self.fps


class CameraVideoStreamTrack(VideoStreamTrack):
    """ 从摄像头获取视频流 """
    def __init__(self, camera_id, width, height):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame_count = 0
        if not os.path.exists(f'../inputs/video_track/{self.id}'):
            print(f'output_folder not exist, creating inputs/video_track/{self.id}......')
            os.makedirs(f'../inputs/video_track/{self.id}')

    async def recv(self):
        self.frame_count += 1
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame from camera")
            return None
        try:
            if self.frame_count % 100 == 1:
                cv2.imwrite(f'../inputs/video_track/{self.id}/send_frame_{self.frame_count}.jpg', frame)
        except Exception as e:
            print(self.id, e)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, 30)  # 30 FPS
        return video_frame


class ExternalVideoStreamTrack(VideoStreamTrack):
    """允许外部动态推送帧，并根据实际时间动态设置时间戳"""
    _start: float
    _timestamp: int

    def __init__(self):
        super().__init__()
        self.frame = None
        self.frame_count = 0
        self.new_frame_event = asyncio.Event()
        self.fps_calculator = FpsCalculator(window_size=30)

    def push_f(self, frame):
        """外部推送视频帧"""
        self.frame = frame
        self.frame_count += 1
        self.fps_calculator.update()
        self.new_frame_event.set()

    async def next_timestamp(self):
        if hasattr(self, "_timestamp"):
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE

    async def recv(self):
        # 等待新帧到达
        await self.new_frame_event.wait()
        self.new_frame_event.clear()
        pts, time_base = await self.next_timestamp()

        if not os.path.exists(f'../inputs/video_track/{self.id}'):
            os.makedirs(f'../inputs/video_track/{self.id}')
        if self.frame_count % 20 == 0:
            print(f'\033[32mfrom sender {self.id}: {self.fps_calculator.get_fps():.1f}fps')

        try:
            if self.frame_count % 100 == 1:
                cv2.imwrite(f'../inputs/video_track/{self.id}/send_frame_{self.frame_count}.jpg', self.frame)
        except Exception as e:
            print(self.id, e)
        
        video_frame = VideoFrame.from_ndarray(self.frame, format="rgb24")
        
        video_frame.pts = pts
        video_frame.time_base = time_base
        
        return video_frame
    

class LoopingVideoStreamTrack(VideoStreamTrack):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = 0
        if not self.cap.isOpened():
            raise Exception(f"Could not open video file {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = 0
        if not os.path.exists(f'../inputs/video_track/{self.id}'):
            os.makedirs(f'../inputs/video_track/{self.id}')
            print(f'creating ../inputs/video_track/{self.id} .......')

    async def recv(self):
        ret, frame = self.cap.read()
        if not ret:
            # When video file ends, rewind to the beginning
            print(f"End of video, rewinding to the beginning...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

        self.frame_count += 1
        try:
            if self.frame_count % 100 == 1:
                cv2.imwrite(f'../inputs/video_track/{self.id}/send_frame_{self.frame_count}.jpg', frame)
        except Exception as e:
            print(self.id, e)
        # Create video frame to send over WebRTC
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self.frame_count              
        video_frame.time_base = fractions.Fraction(1, 30)  # Time base based on FPS
        return video_frame