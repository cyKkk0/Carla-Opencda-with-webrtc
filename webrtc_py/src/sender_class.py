import asyncio
import pickle
import cv2
import re
import os
import random
import time
import threading
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import TcpSocketSignaling
try:
    from src.media import CameraVideoStreamTrack, ExternalVideoStreamTrack, LoopingVideoStreamTrack
except:
    pass

try:
    from media import CameraVideoStreamTrack, ExternalVideoStreamTrack, LoopingVideoStreamTrack
except:
    pass


def run_server(webrtc_server, server_loop):
    asyncio.set_event_loop(server_loop)
    server_loop.run_until_complete(webrtc_server.run())


def extract_port_from_sdp(sdp: str):
    """
    从 SDP 数据中提取端口号。

    Args:
        sdp (str): SDP 数据。

    Returns:
        int | None: 提取的端口号，如果未找到则返回 None。
    """
    # 使用正则表达式匹配 m= 行
    match = re.search(r"m=.*? (\d+)", sdp)
    if match:
        return int(match.group(1))
    return None


class Webrtc_server:
    def __init__(self, ip_address, port):
        self.ip_address = ip_address
        self.port = port
        self.signaling = TcpSocketSignaling(ip_address, port)
        self.pc = RTCPeerConnection()
        self.data_channels = {}  # 存储所有数据通道
        self.video_tracks = {}   # 存储所有视频流
        self.if_connected = False

    async def track_bitrate_monitor(self):
        last_bytes = {}
        last_time = time.time()
        while True:
            stats = await self.pc.getStats()
            # print(stats)
            for report in stats.values():
                if report.type == "outbound-rtp" and report.kind == "video":
                    now = time.time()
                    delta_time = now - last_time
                    if last_bytes.get(report.trackId):
                        delta_bytes = report.bytesSent - last_bytes[report.trackId]
                    else:
                        delta_bytes = report.bytesSent
                    bitrate_kbps = (delta_bytes * 8) / 1000 / delta_time
                    print(f"Track {report.trackId}: {bitrate_kbps:.2f} kbps")

                    last_bytes[report.trackId] = report.bytesSent
                    last_time = now
            await asyncio.sleep(1)


    async def renegotiate_sdp(self):
        if not self.if_connected:
            await self.signaling.connect()
            self.if_connected = True

        print('status: ', await self.pc.getStats())
        """重新协商 SDP 以更新媒体轨道"""
        print("Starting SDP renegotiation...")
        offer = await self.pc.createOffer()
        print('Set local description')
        await self.pc.setLocalDescription(offer)
        print('Sending new SDP offer')
        await self.signaling.send(self.pc.localDescription)
        print("Sent new SDP offer")
        obj = await self.signaling.receive()
        if isinstance(obj, RTCSessionDescription):
            await self.pc.setRemoteDescription(obj)
            # print('port:', extract_port_from_sdp(obj.sdp))
            print("Remote description set")
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
        print('waiting for lock')
        async with self.lock:
            print('get lock')
            # track_id = len(self.video_tracks)
            if source == "camera":
                video_track = CameraVideoStreamTrack(camera_id, width, height)
                # print('I\'m a camera track!')
            elif source == "external":
                video_track = ExternalVideoStreamTrack()
            elif source == 'video_file':
                video_track = LoopingVideoStreamTrack(file_path)
                # print('I\'m from file!')
            else:
                raise ValueError("Invalid source. Use 'camera' or 'external'.")
            
            track_id = video_track.id
            self.video_tracks[track_id] = video_track
            
            self.pc.addTrack(video_track)

            await self.renegotiate_sdp()
            print(f"--- Added video track: {track_id}")

            return video_track, track_id

    # iff necessary?
    def push_frame(self, track_id, frame):
        """ 向指定的外部视频流推送帧 """
        if track_id in self.video_tracks and isinstance(self.video_tracks[track_id], ExternalVideoStreamTrack):
            self.video_tracks[track_id].push_f(frame)
            # print(f"Pushed frame to {track_id}")
        else:
            print(f"Track {track_id} is not an external video track")

    async def add_data_channel(self, label):
        """ 运行时动态添加数据通道 """
        async with self.lock:
            if label in self.data_channels:
                print(f"Data channel {label} already exists.")
                return None, None
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

    async def setup_webrtc_and_run(self, test=False):
        if not test:
            await self.add_data_channel('test1')
            count = 0
            while True:
                await asyncio.sleep(5)
                count += 1
                self.data_channels['test1'].send(pickle.dumps(f'hello {count}'))
        else:
            ex_track, ex_track_id = await asyncio.create_task(self.add_video_track(len(self.video_tracks), source='external'))
            img = []
            for i in range(10):
                img.append(cv2.imread(f'/home/bupt/cykkk/carla&opencda/webrtc_py/test_source/pic/{i}.jpg'))
            count = 0
            cnt = 0
            task = asyncio.create_task(self.track_bitrate_monitor())
            while True:
                random_number = round(random.uniform(0.04, 0.06), 2)
                await asyncio.sleep(random_number)
                count += 1
                for i in range (ex_track_id + 1):
                    self.push_frame(i, img[cnt])
                cnt = (cnt + 1) % 9
        self.running = asyncio.Event()
        await self.running.wait()
        
    async def run(self):
        """ 运行 WebRTC 服务器 """
        self.lock = asyncio.Lock()      # only one track can be added at a time
        await self.setup_webrtc_and_run()
    
    def run_server_in_new_thread(self):
        server_loop = asyncio.new_event_loop()
        thread1 = threading.Thread(target=run_server, args=(self, server_loop))
        thread1.start()
        return thread1, server_loop

def call_back():
    print('I call back!!!')

async def main():
    ip_address = "127.0.0.1"
    port = 8080
    streamer = Webrtc_server(ip_address, port)

    task1 = asyncio.create_task(streamer.run())
    await asyncio.sleep(5)
    # await asyncio.create_task(streamer.add_video_track(len(streamer.video_tracks), source='video_file', file_path='/home/bupt/cykkk/carla&opencda/webrtc_py/exam_video/test1.mp4'))
    # await asyncio.create_task(streamer.add_data_channel('test1'))
    # streamer.data_channels['test1'].call_back = call_back
    # print(id(streamer.data_channels['test1']))

    source = 'camera'
    image = np.load(f'/home/bupt/cykkk/record/{source}_processed_data_frame_100.npy')
    image = image[:, :, :3]
    ex_track, ex_track_id = await asyncio.create_task(streamer.add_video_track(len(streamer.video_tracks), source='external'))
    ex_track, ex_track_id = await asyncio.create_task(streamer.add_video_track(len(streamer.video_tracks), source='external'))
    ex_track, ex_track_id = await asyncio.create_task(streamer.add_video_track(len(streamer.video_tracks), source='external'))
    ex_track, ex_track_id = await asyncio.create_task(streamer.add_video_track(len(streamer.video_tracks), source='external'))
    # ex_track, ex_track_id = await asyncio.create_task(streamer.add_video_track(len(streamer.video_tracks), source='external'))
    # ex_track, ex_track_id = await asyncio.create_task(streamer.add_video_track(len(streamer.video_tracks), source='external'))
    # ex_track, ex_track_id = await asyncio.create_task(streamer.add_video_track(len(streamer.video_tracks), source='external'))
    # ex_track, ex_track_id = await asyncio.create_task(streamer.add_video_track(len(streamer.video_tracks), source='external'))

    # with open('/home/bupt/cykkk/record/lidar_processed_data_frame_100.npy', 'rb') as f:
        # data = f.read()
    # data_array = np.frombuffer(data, dtype=np.float32)
    # await asyncio.create_task(streamer.add_data_channel('test2'))
    img = []
    for i in range(10):
        img.append(cv2.imread(f'/home/bupt/cykkk/carla&opencda/webrtc_py/test_source/pic/{i}.jpg'))
    count = 0
    cnt = 0
    # task3 = asyncio.create_task(streamer.track_bitrate_monitor())
    while True:
        random_number = round(random.uniform(0.04, 0.06), 2)
        await asyncio.sleep(random_number)
        count += 1
        # streamer.data_channels['test1'].send(pickle.dumps(f'hello {count}'))
        # if count > 100:
            # break
        # streamer.push_frame(ex_track_id, img[cnt])
        for key in streamer.video_tracks:
            streamer.push_frame(key, img[cnt])
        cnt = (cnt + 1) % 9
    await task1, task3
    

if __name__ == "__main__":
    asyncio.run(main())
