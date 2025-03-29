import pickle
import carla

# 创建一个 VehicleControl 对象
control = carla.VehicleControl(
    throttle=0.5,
    steer=0.0,
    brake=0.0,
    hand_brake=False,
    reverse=False,
    manual_gear_shift=False,
    gear=0
)
print(dir(control))
print(type(control), control)
# 序列化对象
# serialized_control = pickle.dumps(control)
# print(type(serialized_control), serialized_control)
# data = pickle.loads(serialized_control)
# print(type(data), data)
# 将序列化后的数据保存到文件
# with open('control.pkl', 'wb') as f:
#     f.write(serialized_control)