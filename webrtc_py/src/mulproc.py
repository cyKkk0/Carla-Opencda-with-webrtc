import multiprocessing
import time
import asyncio
import threading
import cv2
import os
import numpy as np
import sys
import pickle
import hashlib
import random
from receiver_class import Webrtc_client
from sender_class import Webrtc_server


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


def recv_from_recv(pipe):
    while True:
        # img_bytes = pipe.recv()
        # img_encoded = np.frombuffer(img_bytes, dtype=np.uint8)
        # img = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
        # print("from recv", img.shape)
        data = pipe.recv()
        print(data)
        # arr = np.frombuffer(data, dtype=np.dtype('f4'))
        # print(arr.shape)
        # arr2 = arr.reshape(int(arr.shape[0] / 4), 4)
        # print(type(arr2), arr2.shape)


if __name__ == "__main__":
    # 创建进程
    folder_path = '/home/bupt/cykkk/carla&opencda/webrtc_py/test_source/pic'
    proc = []
    img = []
    img = get_all_pic(folder_path)
    cnt = 0
    fa_pipe = []
    ch_pipe = []
    for i in range(8080, 8081):
        webrtc_server = Webrtc_server('127.0.0.1', i)
        webrtc_client = Webrtc_client('127.0.0.1', i)
        parent_conn, child_conn = multiprocessing.Pipe()
        recv_fa_conn, recv_ch_conn = multiprocessing.Pipe()
        fa_pipe.append(parent_conn)
        ch_pipe.append(child_conn)
        _ = webrtc_server.run_server_in_new_process(add_data=True, recv_pipe=child_conn, label='test')
        # _ = webrtc_server.run_server_in_new_process(add_video=True, recv_pipe=child_conn)
        _ = webrtc_client.run_client_in_new_process(send_pipe=recv_fa_conn)
    proc = _
    time.sleep(3)
    file_path = '/home/bupt/cykkk/record/1_lidar_processed_data_frame_100.npy'

    data = np.load(file_path)
    thread1 = threading.Thread(target=recv_from_recv, args=(recv_ch_conn,))
    thread1.start()
    # for i in range(len(img)):
    #     _, img_encoded = cv2.imencode('.jpg', img[i])
    #     img[i] = img_encoded.tobytes()
    cnt = 0
    # print(f"send size {sys.getsizeof(pickle.dumps(data))}")
    # control msg
    data = {
            'throttle': 1.0, 
            'steer': 0.005158573854714632, 
            'brake': 0.0, 
            'hand_brake': False, 
            'reverse': False, 
            'manual_gear_shift': False, 
            'gear': 0
            }   
    print(pickle.dumps(data))
    print(pickle.loads(pickle.dumps(data)))
    while True:
        for pipe in fa_pipe:
            pipe.send(data)
        time.sleep(1)
        cnt += 1
        # img_bytes = img[cnt]
        # for pipe in fa_pipe:
        #     pipe.send(img_bytes)
        # cnt = (cnt + 1) % len(img)
        # time.sleep(0.1)
        # break


    proc.join()
