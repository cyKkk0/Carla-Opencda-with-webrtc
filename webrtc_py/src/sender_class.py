import asyncio
import pickle
import cv2
import os
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import TcpSocketSignaling
from media import CameraVideoStreamTrack, ExternalVideoStreamTrack, LoopingVideoStreamTrack


class WebRTCStreamer:
    def __init__(self, ip_address, port):
        self.ip_address = ip_address
        self.port = port
        self.signaling = TcpSocketSignaling(ip_address, port)
        self.pc = RTCPeerConnection()
        self.data_channels = {}  # 存储所有数据通道
        self.video_tracks = {}   # 存储所有视频流
        self.if_connected = False
        self.ready = asyncio.Event()
        self.lock = asyncio.Lock() # only one track can be added at a time

    async def renegotiate_sdp(self):
        if not self.if_connected:
            await self.signaling.connect()
            
            self.if_connected = True

        """重新协商 SDP 以更新媒体轨道"""
        print("Starting SDP renegotiation...")
        offer = await self.pc.createOffer()
        print('Set local description')
        await self.pc.setLocalDescription(offer)
        # print(self.pc.localDescription.sdp)
        print('Sending new SDP offer')
        await self.signaling.send(self.pc.localDescription)
        print("Sent new SDP offer")
        obj = await self.signaling.receive()
        if isinstance(obj, RTCSessionDescription):
            await self.pc.setRemoteDescription(obj)
            # print("Remote description set")
            # senders = self.pc.getSenders()
            # for sender in senders:
            #     print(sender)
            #     print(f"Sender Track ID: {sender.track.id}, Kind: {sender.track.kind}")
        elif obj is None:
            print("Signaling ended")
        else:
            print(obj)

    async def add_video_track(self, track_id, source="camera", camera_id=0, width=640, height=480, file_path = './exam_video/test1.mp4'):
        """ 运行时动态添加视频流（支持摄像头或外部输入） """
        # if track_id in self.video_tracks:
        #     print(f"Track {track_id} already exists.")
        #     return
        # TODO: track_id should be generated inside the function instead of outside
        async with self.lock:
            track_id = len(self.video_tracks)
            if source == "camera":
                video_track = CameraVideoStreamTrack(camera_id, width, height, track_id)
                print('I\'m a camera track!')
            elif source == "external":
                video_track = ExternalVideoStreamTrack(track_id)
            elif source == 'video_file':
                video_track = LoopingVideoStreamTrack(file_path, track_id)
                print('I\'m from file!')
            else:
                raise ValueError("Invalid source. Use 'camera' or 'external'.")

            self.video_tracks[track_id] = video_track
            
            self.pc.addTrack(video_track)

            print(f"--- Added video track: {track_id}")
            
            await self.renegotiate_sdp()
            return video_track, track_id

    # iff necessary?
    def push_frame(self, track_id, frame):
        """ 向指定的外部视频流推送帧 """
        if track_id in self.video_tracks and isinstance(self.video_tracks[track_id], ExternalVideoStreamTrack):
            self.video_tracks[track_id].push_f(frame)
            print(f"Pushed frame to {track_id}")
        else:
            print(f"Track {track_id} is not an external track")

    async def add_data_channel(self, label):
        """ 运行时动态添加数据通道 """
        async with self.lock:
            # if label in self.data_channels:
            #     print(f"Data channel {label} already exists.")
            #     return None, None
            channel = self.pc.createDataChannel(label)
            self.data_channels[label] = channel

            @channel.on("open")
            def on_open():
                print(f"Data channel {label} is open.")

            @channel.on("message")
            def on_message(message):
                print(f"Received on {label}: {message}")

            print(f"Added data channel: {label}")
            await self.renegotiate_sdp()
            return channel, label

    async def setup_webrtc_and_run(self):
        
        await self.add_video_track(len(self.video_tracks))
        
        # await self.add_video_track(len(self.video_tracks), source='video_file', file_path='./exam_video/test1.mp4')
        while True:
            await asyncio.sleep(5) 
        
    async def run(self):
        """ 运行 WebRTC 服务器 """
        await self.setup_webrtc_and_run()

def call_back():
    print('I call back!!!')

async def main():
    ip_address = "127.0.0.1"
    port = 8080
    streamer = WebRTCStreamer(ip_address, port)

    task1 = asyncio.create_task(streamer.run())
    await asyncio.sleep(5)
    # await asyncio.create_task(streamer.add_video_track(len(streamer.video_tracks), source='video_file', file_path='/home/bupt/cykkk/carla&opencda/webrtc_py/exam_video/test1.mp4'))
    await asyncio.create_task(streamer.add_data_channel('test1'))
    streamer.data_channels['test1'].call_back = call_back
    print(id(streamer.data_channels['test1']))
    # await asyncio.create_task(streamer.add_video_track(len(streamer.video_tracks), source='external'))
    # with open('/home/bupt/cykkk/record/lidar_processed_data_frame_100.npy', 'rb') as f:
        # data = f.read()
    # data_array = np.frombuffer(data, dtype=np.float32)
    # await asyncio.create_task(streamer.add_data_channel('test2'))
    # img = cv2.imread('/home/bupt/cykkk/carla&opencda/webrtc_py/test/test.jpg')
    count = 0
    while True:
        await asyncio.sleep(1)
        count += 1
        streamer.data_channels['test1'].send(pickle.dumps(f'hello {count}'))
    #     streamer.data_channels['test2'].send(pickle.dumps(data_array))
        if count > 100:
            break
        # streamer.push_frame(len(streamer.video_tracks)-1, img)
    await task1
    

if __name__ == "__main__":
    asyncio.run(main())
