import sys
import cv2
import os

if __name__=='__main__':
    print(os.getcwd())
    img = cv2.imread('/home/bupt/cykkk/carla&opencda/webrtc_py/test/test.jpg')
    # print(img)
    cv2.imwrite('test_.jpg', img)