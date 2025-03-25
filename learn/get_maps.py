import carla
import cv2

def main():
    try:
        # 连接到CARLA客户端
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)  # 设置超时时间

        # 获取所有可用的地图
        available_maps = client.get_available_maps()
        print("可用的地图列表：")
        for map_name in available_maps:
            print(map_name)

        # 获取当前加载的地图
        world = client.get_world()
        current_map = world.get_map()
        print("当前加载的地图是:", current_map.name)

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    main()