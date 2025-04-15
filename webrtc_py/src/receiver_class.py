import threading
import asyncio
import time
import cv2
import os
from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    MediaStreamTrack,
    RTCDataChannel
)
from collections import deque
from aiortc.contrib.signaling import TcpSocketSignaling, BYE
from av import VideoFrame
import numpy as np
import pickle


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


def run_client(webrtc_client, client_loop):
    asyncio.set_event_loop(client_loop)
    client_loop.run_until_complete(webrtc_client.start())


class VideoReceiver:
    def __init__(self, track, track_id):
        self.track = track
        self.track_id = track_id
        self.call_back2 = None
        self.fps_calc = FpsCalculator(20)

    def set_callback_func(self, callback_func):
        self.call_back2 = callback_func

    def set_weak_self(self, weak_self):
        self.weak_self = weak_self

    async def handle_track(self):
        # print("Inside handle track")
        frame_count = 0
        while True:
            try:
                frame = await asyncio.wait_for(self.track.recv(), timeout=5.0)
                frame_count += 1                
                if isinstance(frame, VideoFrame):
                    self.fps_calc.update()
                    # print(f"Frame type: VideoFrame, pts: {frame.pts}, time_base: {frame.time_base}")
                    frame = frame.to_ndarray(format="rgb24")
                    if frame_count % 20 == 0:
                        print(f'from recv {self.track_id}: {self.fps_calc.get_fps():.1f}fps')
                    # frame = frame.to_ndarray(format="yuv420p")
                    # frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
                elif isinstance(frame, np.ndarray):
                    print(f"Frame type: numpy array")
                else:
                    print(f"Unexpected frame type: {type(frame)}")
                    continue
                if self.call_back2:
                    if self.catg == 'Camera':
                        self.call_back2(weak_self=self.weak_self, image=frame)
                    elif self.catg == 'test':
                        self.call_back2()
                # if not os.path.exists(f'../outputs/video_track/{self.track_id}'):
                #     os.makedirs(f'../outputs/video_track/{self.track_id}')
                # try:
                #     if frame_count % 50 == 1:
                #         cv2.imwrite(f"../outputs/video_track/{self.track_id}/received_frame_{frame_count}.jpg", frame)
                # except Exception as e:
                #     print(e)
            except asyncio.TimeoutError:
                print(f"{self.track_id} Timeout waiting for frame, continuing...")
            except Exception as e:
                # print(self.track_id)
                print(f"Error in handle_track: {str(e)}")
                break
        print("Exiting handle_track")

class Webrtc_client:
    def __init__(self, ip_address, port):
        self.ip_address = ip_address
        self.port = port
        self.signaling = TcpSocketSignaling(ip_address, port)
        self.pc = RTCPeerConnection()
        self.video_tracks = {}  # 存储所有视频流轨道
        self.data_channels = {}  # 存储所有数据通道

    async def start(self):
        # 连接到信令服务器
        await self.signaling.connect()
        # 配置 RTCPeerConnection 的事件处理
        @self.pc.on("track")
        def on_track(track):
            # if isinstance(track, MediaStreamTrack):
                print(f"Receiving {track.kind} track")
                self.video_tracks[track.id] = VideoReceiver(track, track.id)
                asyncio.ensure_future(self.video_tracks[track.id].handle_track())

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            self.data_channels[channel.label] = channel
            
            print(f"Data channel {channel.label} created.")
            @channel.on("message")
            def on_message(message):
                # self.data_channels[channel.label].on_message(message)
                if hasattr(self.data_channels[channel.label], 'call_back_2'):
                    if self.data_channels[channel.label].cate == 'Lidar':
                        self.data_channels[channel.label].call_back_2(self.data_channels[channel.label].weak_self, pickle.loads(message))
                    else:
                        print(f"Received on {channel.label}: {pickle.loads(message)}")   
                else:
                    print(f"Received on {channel.label}: {pickle.loads(message)}")
            
        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state is {self.pc.connectionState}")

        print("Waiting for offer from sender...")
        offer = await self.signaling.receive()
        # print(f'type: {type(offer)}')
        print("Offer received")
        await self.pc.setRemoteDescription(offer)
        print("Remote description set")

        answer = await self.pc.createAnswer()
        print("Answer created")
        await self.pc.setLocalDescription(answer)
        print("Local description set")

        # print("Local SDP:")
        # print(self.pc.localDescription.sdp)

        await self.signaling.send(self.pc.localDescription)
        print("Answer sent to sender")

        print("Waiting for connection to be established...")
        while self.pc.connectionState != "connected":
            await asyncio.sleep(0.1)

        while True:
            obj = await self.signaling.receive()

            if isinstance(obj, RTCSessionDescription):
                await self.pc.setRemoteDescription(obj)
                answer = await self.pc.createAnswer()
                print("Answer created")
                await self.pc.setLocalDescription(answer)
                print("Local description set")
                await self.signaling.send(self.pc.localDescription)
                print("Answer sent to sender")
            elif isinstance(obj, RTCIceCandidate):
                await self.pc.addIceCandidate(obj)
            elif obj is BYE:
                print("Exiting")
                break

    def run_client_in_new_thread(self):
        time.sleep(3)
        client_loop = asyncio.new_event_loop()
        thread1 = threading.Thread(target=run_client, args=(self,client_loop))
        thread1.start()
        return thread1, client_loop


async def main():
    ip_address = "127.0.0.1"
    port = 8080
    receiver = Webrtc_client(ip_address, port)
    await receiver.start()

if __name__ == "__main__":
    asyncio.run(main())