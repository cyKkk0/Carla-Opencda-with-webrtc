import asyncio
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
from aiortc.contrib.signaling import TcpSocketSignaling, BYE
from av import VideoFrame
import numpy as np
import pickle

def call_back():
    print('I call back!')

class VideoReceiver:
    def __init__(self, track, track_id):
        self.track = track
        self.track_id = track_id
        self.call_back = None

    def set_callback_func(self, callback_func):
        self.call_back = callback_func

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
                    # print(f"Frame type: VideoFrame, pts: {frame.pts}, time_base: {frame.time_base}")
                    frame = frame.to_ndarray(format="bgr24")
                elif isinstance(frame, np.ndarray):
                    print(f"Frame type: numpy array")
                else:
                    print(f"Unexpected frame type: {type(frame)}")
                    continue
                if self.call_back:
                    self.call_back(weak_self=self.weak_self, frame=frame)

                if not os.path.exists(f'../outputs/video_track/{self.track_id}'):
                    os.makedirs(f'../outputs/video_track/{self.track_id}')
                if frame_count % 100 == 1:
                    cv2.imwrite(f"../outputs/video_track/{self.track_id}/received_frame_{frame_count}.jpg", frame)
    
            except asyncio.TimeoutError:
                print("Timeout waiting for frame, continuing...")
            except Exception as e:
                print(self.track_id)
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
            if isinstance(track, MediaStreamTrack):
                print(f"Receiving {track.kind} track")
                self.video_tracks[track.id] = VideoReceiver(track, track.id)
                asyncio.ensure_future(self.video_tracks[track.id].handle_track())

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            channel.call_back = call_back
            self.data_channels[channel.label] = channel
            
            print(f"Data channel {channel.label} created.")
            @channel.on("message")
            def on_message(message):
                # self.data_channels[channel.label].on_message(message)
                if hasattr(self.data_channels[channel.label], 'call_back_2'):
                    self.data_channels[channel.label].call_back_2(self.data_channels[channel.label].weak_self, pickle.loads(message))
                else:
                    print(f"Received on {channel.label}: {pickle.loads(message)}")
            
        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state is {self.pc.connectionState}")
            if self.pc.connectionState == "connected":
                print("WebRTC connection established successfully")

        print("Waiting for offer from sender...")
        offer = await self.signaling.receive()
        print(f'type: {type(offer)}')
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

async def main():
    ip_address = "127.0.0.1"
    port = 8080
    receiver = Webrtc_client(ip_address, port)
    await receiver.start()

if __name__ == "__main__":
    asyncio.run(main())