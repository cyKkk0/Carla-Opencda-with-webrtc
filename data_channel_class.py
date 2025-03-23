from aiortc import RTCIceServer, RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import BYE
from aiortc import rtcdatachannel
import json
import pickle

class CustomDataChannel(rtcdatachannel.RTCDataChannel):
    def __init__(self, label):
        super().__init__(label)
        
    def send_custom_message(self, data):
        # 序列化
        message = pickle.dumps(data)
        # 发送消息
        self.send(message)
    
    def on_message(self, msg):
        try:
            data = pickle.loads(msg)  # 解析 JSON 数据
            print(f"Label {self.label} Received custom message: {data}")
        except Exception as e:
            print(f"Received error {e}")
