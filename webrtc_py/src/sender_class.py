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


def run_server(webrtc_server, server_loop, test, fps, compressed):
    asyncio.set_event_loop(server_loop)
    server_loop.run_until_complete(webrtc_server.run(test, fps, compressed))


def get_all_pic(folder_path):
    if not os.path.exists(folder_path):
        print(f'{folder_path} not exists')
        return []
    res = []
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(folder_path):
        input_path = os.path.join(folder_path, filename)
        if os.path.isfile(input_path):
            # 获取文件扩展名
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                res.append(cv2.imread(input_path))
    return res  



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

        """重新协商 SDP 以更新媒体轨道"""
        # print("Starting SDP renegotiation...")
        offer = await self.pc.createOffer()
        # print('Set local description')
        await self.pc.setLocalDescription(offer)
        # print('Sending new SDP offer')
        await self.signaling.send(self.pc.localDescription)
        # print("Sent new SDP offer")
        obj = await self.signaling.receive()
        if isinstance(obj, RTCSessionDescription):
            await self.pc.setRemoteDescription(obj)
        elif obj is None:
            print("Signaling ended")
        else:
            print(obj)

    async def add_video_track(self, source="camera", camera_id=0, width=640, height=480, file_path = './exam_video/test1.mp4'):
        """ 运行时动态添加视频流（支持摄像头或外部输入） """
        # print('waiting for lock')
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

    async def setup_webrtc_and_run(self, test, fps, folder_path):
        if not test:
            await self.add_data_channel('test1')
            count = 0
            while True:
                await asyncio.sleep(5)
                count += 1
                self.data_channels['test1'].send(pickle.dumps(f'hello {count}'))
        else:
            for i in range(5):
                ex_track, ex_track_id = await asyncio.create_task(self.add_video_track(source='external'))
            img = []
            img = get_all_pic(folder_path)
            count = 0
            cnt = 0
            # task = asyncio.create_task(self.track_bitrate_monitor())
            while True:
                random_number = fps
                # random_number = round(random.uniform(fps-0.01, fps+0.01), 2)
                await asyncio.sleep(random_number)
                count += 1
                for key in self.video_tracks:
                    self.push_frame(key, img[cnt])
                cnt = (cnt + 1) % len(img)
        self.running = asyncio.Event()
        await self.running.wait()
        
    async def run(self, test=False, fps=0.05, folder_path=''):
        """ 运行 WebRTC 服务器 """
        self.lock = asyncio.Lock()      # only one track can be added at a time
        await self.setup_webrtc_and_run(test, fps, folder_path)
    
    def run_server_in_new_thread(self, test=False, fps=0.05, folder_path=''):
        server_loop = asyncio.new_event_loop()
        thread1 = threading.Thread(target=run_server, args=(self, server_loop, test, fps, folder_path))
        thread1.start()
        return thread1, server_loop

def call_back():
    print('I call back!!!')

async def main():
    ip_address = "127.0.0.1"
    port = 8080
    streamer = Webrtc_server(ip_address, port)
    folder_path = os.path.join(os.path.dirname(os.getcwd()), 'test_source/pic')
    print(folder_path)
    task1 = asyncio.create_task(streamer.run(test=True, folder_path=folder_path))
    await asyncio.sleep(5)

    await task1
    

if __name__ == "__main__":
    asyncio.run(main())
