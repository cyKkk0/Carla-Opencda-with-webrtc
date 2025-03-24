import time

class Sensor:
    def __init__(self):
        self._listeners = []  # 存储回调函数列表

    def listen(self, callback):
        """ 注册回调函数 """
        self._listeners.append(callback)  # 将回调函数添加到监听器列表中

    def _trigger_listeners(self, data):
        """ 当传感器数据到达时，触发回调函数 """
        for listener in self._listeners:
            listener(data)  # 依次调用注册的回调函数

    def start_data_stream(self):
        """ 模拟传感器数据流，每隔2秒生成数据并触发回调 """
        while True:
            time.sleep(2)
            data = {"accelerometer": (1.0, 2.0, 3.0)}  # 模拟数据
            self._trigger_listeners(data)  # 触发回调

# 回调函数
def process_data(data):
    print(f"Processing sensor data: {data}")

# 创建传感器实例
sensor = Sensor()

# 注册回调函数
sensor.listen(process_data)

# 启动传感器模拟数据流
sensor.start_data_stream()
