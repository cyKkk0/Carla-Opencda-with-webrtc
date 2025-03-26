import asyncio
import cv2
import os
from aiortc import VideoStreamTrack
from av import VideoFrame
import fractions
import numpy as np

class CameraVideoStreamTrack(VideoStreamTrack):
    """ 从摄像头获取视频流 """
    def __init__(self, camera_id, width, height, track_id):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame_count = 0
        self.track_id = track_id
        if not os.path.exists(f'../inputs/video_track/{self.track_id}'):
            print(f'output_folder not exist, creating inputs/video_track/{self.track_id}......')
            os.makedirs(f'../inputs/video_track/{self.track_id}')

    async def recv(self):
        self.frame_count += 1
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame from camera")
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'../inputs/video_track/{self.track_id}/send_frame_{self.frame_count}.jpg', frame)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, 30)  # 30 FPS
        return video_frame

class ExternalVideoStreamTrack(VideoStreamTrack):
    """ 允许外部动态推送帧 """
    def __init__(self, track_id):
        super().__init__()
        self.track_id = track_id
        self.frame = None
        self.frame_count = 0
        self.new_frame_event = asyncio.Event()

    def push_f(self, frame):
        """ 外部推送视频帧 """
        self.frame = frame
        self.frame_count += 1
        self.new_frame_event.set()

    async def recv(self):
        await self.new_frame_event.wait()
        print('--- ex sending')
        if not os.path.exists(f'../inputs/video_track/{self.track_id}'):
            os.makedirs(f'../inputs/video_track/{self.track_id}')
        cv2.imwrite(f'../inputs/video_track/{self.track_id}/send_frame_{self.frame_count}.jpg', self.frame)
        video_frame = VideoFrame.from_ndarray(self.frame, format="rgb24")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, 30)
        return video_frame
    

class LoopingVideoStreamTrack(VideoStreamTrack):
    def __init__(self, video_path, track_id):
        super().__init__()
        self.video_path = video_path
        self.track_id = track_id
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = 0
        if not self.cap.isOpened():
            raise Exception(f"Could not open video file {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f'fps: {self.fps}')
        self.frame_count = 0
        if not os.path.exists(f'../inputs/video_track/{self.track_id}'):
            os.makedirs(f'../inputs/video_track/{self.track_id}')
            print(f'creating ../inputs/video_track/{self.track_id} .......')

    async def recv(self):
        ret, frame = self.cap.read()
        if not ret:
            # When video file ends, rewind to the beginning
            print(f"End of video, rewinding to the beginning...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

        self.frame_count += 1
        cv2.imwrite(f'../inputs/video_track/{self.track_id}/send_frame_{self.frame_count}.jpg', frame)

        # Create video frame to send over WebRTC
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self.frame_count              
        video_frame.time_base = fractions.Fraction(1, 30)  # Time base based on FPS
        # video_frame.time_base = fractions.Fraction(1, self.fps)  # will block on this line, why??
        return video_frame