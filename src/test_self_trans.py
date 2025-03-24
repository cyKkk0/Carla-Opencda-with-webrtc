import os, sys
import asyncio

from sender_class import WebRTCStreamer
from receiver_class import WebRTCReceiver


async def main():
    ip = '127.0.0.1'
    port = 8080    
    server = WebRTCStreamer(ip, port)
    client = WebRTCReceiver(ip, port)
    asyncio.create_task(server.run())
    server.ready.wait()
    await asyncio.sleep(1)
    await client.start()


if __name__ == '__main__':
    asyncio.run(main())
