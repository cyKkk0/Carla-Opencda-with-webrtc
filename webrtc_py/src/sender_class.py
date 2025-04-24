import asyncio
import pickle
import cv2
import re
import os
import random
import time
import threading
import multiprocessing
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


def run_server(webrtc_server, test, add_video, add_data, recv_pipe, label):
    server_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(server_loop)
    server_loop.run_until_complete(webrtc_server.run(test, add_video, add_data, recv_pipe, label))


def get_all_pic(folder_path):
    """
    load all pics in the folder_path/
    """
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
    # not so accurate
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
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        await self.signaling.send(self.pc.localDescription)
        obj = await self.signaling.receive()
        if isinstance(obj, RTCSessionDescription):
            await self.pc.setRemoteDescription(obj)
        elif obj is None:
            print("Signaling ended")
        else:
            print(obj)

    async def add_video_track(self, source="camera", camera_id=0, width=640, height=480, file_path = './exam_video/test1.mp4'):
        """ 运行时动态添加视频流（支持摄像头或外部输入） """
        async with self.lock:
            if source == "camera":
                video_track = CameraVideoStreamTrack(camera_id, width, height)
            elif source == "external":
                video_track = ExternalVideoStreamTrack()
            elif source == 'video_file':
                video_track = LoopingVideoStreamTrack(file_path)
            else:
                raise ValueError("Invalid source. Use 'camera' or 'external'.")
            
            track_id = video_track.id
            self.video_tracks[track_id] = video_track
            
            self.pc.addTrack(video_track)

            await self.renegotiate_sdp()
            print(f"--- Added video track: {track_id}")
            return video_track, track_id

    def push_frame(self, track_id, frame):
        """ 向指定的外部视频流推送帧 """
        if track_id in self.video_tracks and isinstance(self.video_tracks[track_id], ExternalVideoStreamTrack):
            self.video_tracks[track_id].push_f(frame)
            # print(f"Pushed frame to {track_id}")
        else:
            print(f"Track {track_id} is not an external video track")

    def push_data(self, label, data):
        if label in self.data_channels:
            try:
                # you need to convert your data to bytes
                self.data_channels[label].send(data)
            except Exception as e:
                print(e)

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

    async def setup_webrtc_and_run(self, test, add_video, add_data, recv_pipe, label):
        if add_video and add_data:
            print('you can only add track or channel at the same time')
        elif add_video:
            if not recv_pipe:
                print("to run in new process, a recv pipe is needed")
                return
            ex_track, ex_track_id = await asyncio.create_task(self.add_video_track(source='external'))
            loop = asyncio.get_event_loop()
            # convert sync to async by loop.run_in_executor or other implementation like asyncio.Queue or aiozmq?
            while True:
                img_bytes = await loop.run_in_executor(
                    None,  # 使用默认线程池
                    recv_pipe.recv
                )
                img_encoded = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
                self.push_frame(ex_track_id, img)
        elif add_data:
            if not recv_pipe:
                print("to run in new process, a recv pipe is needed")
                return
            if not label:
                label = 'admin'
                print("no label config, use default label 'admin'")
            await self.add_data_channel(label)
            loop = asyncio.get_event_loop()
            # convert sync to async by loop.run_in_executor or other implementation like asyncio.Queue or aiozmq?
            while True:
                data = await loop.run_in_executor(
                    None,  # 使用默认线程池
                    recv_pipe.recv
                )
                # load data and then send or send directly?
                self.push_data(label, data)
        elif test:
            _, ex_track_id = await asyncio.create_task(self.add_video_track(source='external'))
            frame = np.full((600, 800, 3), (255, 255, 255), dtype=np.uint8)
            while True:
                await asyncio.sleep(1) # 1 fps
                self.push_frame(ex_track_id, frame)
        else:
            print("no configuration, to maintain the server, add a data channel labeled 'admin'")
            await self.add_data_channel('admin')
            self.running = asyncio.Event()
            await self.running.wait()
        
    async def run(self, test=False, add_video=False, add_data=False, recv_pipe=None, label=''):
        """ 运行 WebRTC 服务器 """
        self.lock = asyncio.Lock()      # only one track can be added at a time
        await self.setup_webrtc_and_run(test, add_video, add_data, recv_pipe, label)
    
    def run_server_in_new_process(self, test=False, add_video=False, add_data=False, recv_pipe=None, label=''):
        proc1 = multiprocessing.Process(target=run_server, args=(self, test, add_video, add_data, recv_pipe, label))
        proc1.start()
        return proc1
    

async def main():
    pass
    

if __name__ == "__main__":
    asyncio.run(main())
