import os, sys
import asyncio

from src.sender_class import WebRTCStreamer
from src.receiver_class import WebRTCReceiver


async def main():
    ip = '127.0.0.1'
    port = 8080    
    server = WebRTCStreamer(ip, port)
    client = WebRTCReceiver(ip, port)
    server_task = asyncio.create_task(server.run())
    await server.ready.wait()
    await client.start()


if __name__ == '__main__':
    asyncio.create_task(main().run())
