import cv2
import time

def test(a = 2):
    print(a)

def test1():
    cap = cv2.VideoCapture(0)  # 打开默认摄像头
    width = 960
    height = 540
    if cap.isOpened():
        print("Supported resolutions:")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if w == width and h == height:
            print(f"Width: {width}, Height: {height}")
        cap.release()
    else:
        print("Cannot open camera")


if __name__ == '__main__':
    test(5)
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # 获取摄像头的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("摄像头的帧率是：", fps)

    # 释放摄像头
    cap.release()
