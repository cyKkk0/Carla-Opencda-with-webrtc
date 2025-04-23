import time
import asyncio
import threading
import cv2
import random
from receiver_class import Webrtc_client
from sender_class import Webrtc_server


def run_server(webrtc_server, server_loop):
    # server_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(server_loop)
    server_loop.run_until_complete(webrtc_server.run())


def run_client(webrtc_client, client_loop):
    # client_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(client_loop)
    client_loop.run_until_complete(webrtc_client.start())


if __name__ == '__main__':
    thread = []
    end_port = 8084
    webrtc_server = None
    webrtc_client = None
    # folder_path = '/home/bupt/cykkk/carla&opencda/webrtc_py/test_source/pic/compressed'
    folder_path = '/home/bupt/cykkk/carla&opencda/webrtc_py/test_source/pic'
    for i in range(8080, end_port):
        webrtc_server = Webrtc_server('127.0.0.1', i)
        webrtc_client = Webrtc_client('127.0.0.1', i)
        _, server_loop = webrtc_server.run_server_in_new_thread(test=True, fps=0.05,folder_path=folder_path)
        _, client_loop = webrtc_client.run_client_in_new_thread()
        thread.append(_)
    time.sleep(5)
    # future = asyncio.run_coroutine_threadsafe(test(webrtc_server), server_loop)
    # future.result()
    thread[0].join()