import asyncio
import cv2
import os
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCRtpCodecParameters, \
                RTCOutboundRtpStreamStats, RTCTransportStats
from aiortc.contrib.signaling import TcpSocketSignaling
from av import VideoFrame
import fractions
from datetime import datetime

class CustomVideoStreamTrack(VideoStreamTrack):
    def __init__(self, camera_id, width, height):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame_count = 0

    async def recv(self):
        self.frame_count += 1
        # print(f"Sending frame {self.frame_count}")
        ret, frame = self.cap.read()    
        if not os.path.exists('inputs'):
            os.mkdir('inputs')
        cv2.imwrite(f'./inputs/send_frame_{self.frame_count}.jpg', frame)
        # print(frame.shape)
        if not ret:
            print("Failed to read frame from camera")
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, 30)  # Use fractions for time_base
        # Add timestamp to the frame
        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Current time with milliseconds
        # cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return video_frame

async def monitor_send_network_metrics(pc):
    print("running")
    # while True:
        # senders = pc.getSenders()
        # print(senders)
        # for sender in senders:
            # stats = await sender.getStats()
            # 遍历统计信息
            # for stat in stats.values():
            #     if isinstance(stat, RTCOutboundRtpStreamStats):
            #         # 实时比特率（bps）：通过 bytesSent 和时间戳计算
            #         # 这里假设时间间隔为1秒，实际应用中需要根据时间戳计算
            #         print("实时比特率（bps）:", stat.bytesSent * 8)
            #         # 发送帧率（fps）
            #         print("发送帧率（fps）:", stat.framesSent)
            #         # 发送延迟（ms）：roundTripTime
            #         print("发送延迟（ms）:", stat.roundTripTime)
            #     elif isinstance(stat, RTCTransportStats):
            #         # 发送的数据包数
            #         print("发送的数据包数:", stat.packetsSent)
            #         # 重传的数据包数
            #         print("重传的数据包数:", stat.packetsRetransmitted)
        # await asyncio.sleep(1)

async def setup_webrtc_and_run(ip_address, port, camera_id, width, height):
    signaling = TcpSocketSignaling(ip_address, port)
    pc = RTCPeerConnection()
    # pc = RTCPeerConnection(
    #     configuration=RTCConfiguration(
    #         codecs=[
    #             RTCRtpCodecParameters(
    #                 mimeType="video/H264",
    #                 payloadType=100,
    #                 clockRate=90000,
    #                 sdpFmtpLine="profile-level-id=42e01f;level-asymmetry-allowed=1;packetization-mode=1"
    #             )
    #         ]
    #     )
    # )
    video_sender = CustomVideoStreamTrack(camera_id, width, height)
    sender = pc.addTrack(video_sender)
    # print(sender.getParameters())
    try:
        await signaling.connect()

        @pc.on("datachannel")
        def on_datachannel(channel):
            print(f"Data channel established: {channel.label}")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state is {pc.connectionState}")
            if pc.connectionState == "connected":
                print("WebRTC connection established successfully")

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        print("Local SDP:")
        print(pc.localDescription.sdp)
        await signaling.send(pc.localDescription)

        print("running")
        asyncio.ensure_future(monitor_send_network_metrics(pc))

        while True:
            obj = await signaling.receive()
            if isinstance(obj, RTCSessionDescription):
                await pc.setRemoteDescription(obj)
                print("Remote description set")
            elif obj is None:
                print("Signaling ended")
                break
        print("Closing connection")
    finally:
        await pc.close()

async def main(width = 640, height = 480):
    ip_address = "127.0.0.1"
    port = 8080
    camera_id = 0
    await setup_webrtc_and_run(ip_address, port, camera_id, width, height)

if __name__ == "__main__":
    width, height = 960, 540
    asyncio.run(main(width, height))