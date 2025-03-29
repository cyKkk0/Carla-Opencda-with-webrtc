import os
import sys
import time
import asyncio
import threading
import pickle
from omegaconf import OmegaConf
from sender_class import Webrtc_server
from receiver_class import Webrtc_client

async def start_server(webrtc_server, webrtc_client):
    asyncio.create_task(webrtc_server.run())
    await asyncio.sleep(3)
    await webrtc_client.start()


async def run_with_webrtc(webrtc_server, webrtc_client):
    try:
        task = asyncio.create_task(start_server(webrtc_server, webrtc_client))
        await task
    except Exception as e:
        print(e)

def create_data(webrtc_server):
    return

async def add_track(webrtc_server, webrtc_client):
    try:
        for i in range(1, 10):
            pass
        await webrtc_server.add_data_channel('test')
        count = 0
        while True:
            await asyncio.sleep(1)
            count += 1
            webrtc_server.data_channels['test'].send(pickle.dumps(f'hello {count}'))
    except:
        return


async def main():
    webrtc_server = Webrtc_server('127.0.0.1', 8080)
    webrtc_client = Webrtc_client('127.0.0.1', 8080)
    task1 = asyncio.create_task(run_with_webrtc(webrtc_server, webrtc_client))
    await asyncio.sleep(5)
    task2 = asyncio.create_task(add_track(webrtc_server, webrtc_client))
    await task1, task2
        # task1 = asyncio.create_task(run_webrtc(webrtc_server, webrtc_client))
        # asyncio.run(run_webrtc(webrtc_server, webrtc_client))


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(' - Exited by user.')
