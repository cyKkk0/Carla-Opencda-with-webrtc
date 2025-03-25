import asyncio


async def hello():
    print('hello')
    return 666


async def main():
    task1 = asyncio.create_task(hello())
    a = await task1
    print(a)

if __name__=='__main__':
    asyncio.run(main())
